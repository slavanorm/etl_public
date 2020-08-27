import pandas as pd
import numpy as np
from pandas import DataFrame as d
from datetime import timedelta
from tenacity import retry, wait, wait_fixed
from main.base import (
    Extract,
    Transform,
    Load,
    pretty,
    populate_kwargs,
    df_filter,
    pd_list_unique,
    today,
    year_start,
)
from time import sleep
import json


class GC_e(Extract):
    def __init__(self):
        super().__init__(
            url="https://edu.neurosofia.ru/pl/api/account/",
            v="1.0",
            vk_group_id="120358198",
        )
        self.key = json.load(
            open("../access/gc.json", "r")
        )

    def get(
        self,
        endpoint: str = "",
        data: dict = None,
        write: bool = False,
        **kwargs,
    ):
        return super().extract(
            endpoint=endpoint,
            method="POST",
            data={"key": self.key},
            **kwargs,
        )

    def extract(
        self,
        task: str = "users",
        date_from: str = None,
        date_to: str = None,
        status: str = None,
        export_id: int = None,
        from_pickle: bool = False,
        write: bool = False,
    ):
        """
        Args:
            export_id: could b received from error
            task: ['users','deals']
            date_from: 'YYYY-MM-DD'
            date_to: 'YYYY-MM-DD'
            from_pickle: bool
            status: {users:[active,in_base],
            write: bool
        """

        status_list = [
            "new",
            "payed",
            "cancelled",
            "in_work",
            "payment_waiting",
            "part_payed",
            "waiting_for_return",
            "not_confirmed",
            "pending",
        ]

        def check_args():

            if task not in ["users", "deals"]:
                raise TypeError
            if task == "users":
                if status not in [
                    None,
                    "active",
                    "in_base",
                ]:
                    raise TypeError
            if task == "deals":
                if status not in [
                    None,
                    "new",
                    "payed",
                    "cancelled",
                    "in_work",
                    "payment_waiting",
                    "part_payed",
                    "waiting_for_return",
                    "not_confirmed",
                    "pending",
                ]:
                    raise TypeError

        fname = f"../exports/GC_{task}"
        if status is not None:
            fname += "_" + status
        fname += "_extracted.p"
        if from_pickle:
            return Extract.read(fname)

        @retry(wait=wait_fixed(30))
        def get_export_id():
            ans = self.get(endpoint=task, **kwargs)
            if "error_code" in ans:
                print(ans["error_message"])
                raise
            export_id = ans["info"]["export_id"]
            return export_id

        @retry(wait=wait_fixed(200))
        def get_export(export_id):

            ans = self.get(
                endpoint="exports/" + str(export_id),
            )
            if "error_code" in ans:
                print(ans["error_message"])
                raise
            return ans

        def date_check(data, date: str):
            data = data["info"]
            check = data["fields"].index("Создан")
            check = data["items"][-1][check]
            check = check[:10]
            return check

        def merge_json(ans1: dict, ans2: dict):
            data = ans1["info"]
            col = data["fields"].index("Создан")
            check = data["items"][-1][col]
            ans1["info"]["items"] = [
                e
                for e in ans1["info"]["items"]
                if not e[col].startswith(check)
            ]
            ans1["info"]["items"].append(
                ans2["info"]["items"]
            )
            return ans1

        check_args()
        kwargs = populate_kwargs(
            ("created_at[to]", date_to, today),
            (
                "created_at[from]",
                date_from,
                year_start,
            ),
            ("status", status),
        )
        if export_id is None:
            export_id = get_export_id()
            sleep(30)
        ans = get_export(export_id)
        # export <= 50k of rows
        if len(ans["info"]["items"]) > 49997:
            date_from_new = date_check(
                data=ans, date=date_to
            )
            ans2 = self.extract(
                task=task,
                date_from=date_from_new,
                date_to=date_to,
                status=status,
                export_id=export_id,
            )
            if (
                ans2["info"]["fields"]
                != ans["info"]["fields"]
            ):
                raise
            else:
                ans = merge_json(ans, ans2)
        if write:
            Extract.write(fname, ans)
        return ans


class GC_t(Transform):
    cols_naming = [
        "Номер",
        "Пользователь",
        "Email",
        "Телефон",
        "Дата создания",
        "Дата оплаты",
        "Title",
        "Статус",
        "Стоимость, RUB",
        "Оплачено",
        "Получено",
        "Валюта",
        "Город",
        "Платежная система",
        "ID партнера",
        "ФИО партнера пользователя",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_content",
        "user_utm_source",
        "user_utm_medium",
        "user_utm_campaign",
        "user_utm_content",
        "date_mapped",
    ]
    date_out_format = "%Y-%m-%d %H:%M:%S"
    date_mapped_out_format = "%Y-%m-%d"
    date_in_format = "%Y-%m-%d %H:%M:%S"
    cols_mapping = [
        {
            "col": "НТ_mapped",
            "check": [
                "contains",
                "Title",
                "ейротрансф",
            ],
            "result": "HT",
        },
        {
            "col": "НС_mapped",
            "check": ["contains", "Title", "ейрософ"],
            "result": "HC",
        },
        {
            "col": "НС_mapped",
            "check": ["contains", "Title", "нтенсив"],
            "result": "HC",
        },
        {
            "col": "Площадка_mapped",
            "check": [
                "contains",
                "user_utm_source",
                ["vk", "away.vk.com", "vkgroup"],
            ],
            "result": "vk",
        },
        {
            "col": "Площадка_mapped",
            "check": [
                "contains",
                "user_utm_source",
                [
                    "instagram[a-zA-Z]*",
                    "facebook",
                    "marathon",
                ],
            ],
            "result": "ig",
        },
        {
            "col": "Площадка_mapped",
            "check": [
                ["isna", "Площадка_mapped"],
                ["eq", "user_utm_medium", "account"],
            ],
            "result": "ig",
        },
        {
            "col": "Площадка_mapped",
            "check": ["neq", "ID партнера", ""],
            "result": "partner",
        },
        {
            "col": "Площадка_mapped",
            "check": [
                ["isna", "Площадка_mapped"],
                [
                    "contains",
                    "utm_source",
                    [
                        "instagram[a-zA-Z]*",
                        "facebook",
                        "marathon",
                    ],
                ],
            ],
            "result": "ig",
        },
        {
            "col": "Площадка_mapped",
            "check": [
                ["isna", "Площадка_mapped"],
                [
                    "contains",
                    "utm_source",
                    ["vk", "away.vk.com", "vkgroup"],
                ],
            ],
            "result": "vk",
        },
        {
            "col": "Площадка_mapped",
            "check": [
                ["isna", "Площадка_mapped"],
                ["neq", "user_utm_content", ""],
                ["eq", "utm_source", ""],
            ],
            "result": "ig",
        },
        {
            "col": "Площадка_mapped",
            "check": [
                ["isna", "Площадка_mapped"],
                ["eq", "utm_source", "google"],
            ],
            "result": "google",
        },
        {
            "col": "Площадка_mapped",
            "check": [
                ["isna", "Площадка_mapped"],
                ["eq", "user_utm_source", "google"],
            ],
            "result": "google",
        },
        {
            "col": "Площадка_mapped",
            "check": ["isna", "Площадка_mapped"],
            "result": "others",
        },
        {
            "col": "HC_HT_mapped",
            "check": [
                ["eq", "НТ_mapped", "HT"],
                ["neq", "НС_mapped", "HC"],
            ],
            "result": "HT",
        },
        {
            "col": "HC_HT_mapped",
            "check": [
                ["eq", "НС_mapped", "HC"],
                ["neq", "НТ_mapped", "HT"],
            ],
            "result": "HC",
        },
        {
            "col": "HC_HT_mapped",
            "check": [
                ["eq", "НС_mapped", "HC"],
                ["eq", "НТ_mapped", "HT"],
            ],
            "result": "HCHT",
        },
    ]

    @staticmethod
    def date_mapping_func(cell):
        weekday = cell.weekday()
        weekstart = cell.normalize() - timedelta(
            days=weekday
        )
        diff0 = cell - weekstart
        diff = (
            12
            if diff0 > timedelta(days=5, hours=11)
            else 5
        )
        diff = timedelta(days=diff)

        retval = weekstart + diff
        return retval

    def change(self, data: dict):
        datemap = (
            "Дата создания"  # col used for date_mapped
        )

        data = pd.DataFrame.from_records(
            data["info"]["items"],
            columns=data["info"]["fields"],
        )

        if (
            len(pd_list_unique(data["Дата создания"]))
            > 1
        ):
            data = data[data["Дата оплаты"] != ""]
            data = data[
                data["Дата оплаты"].str[:4] != "2019"
            ]
            data["date_mapped"] = pd.to_datetime(
                data[datemap]
            ).apply(self.date_mapping_func)

        return data


def make_pivot(dont_push: bool = True):
    Ge = GC_e()
    df1 = Extract.read("../exports/GC_deals_payed.p")

    sumcol = "Оплачено"
    cols = [
        "HC_HT_mapped",
        "date_mapped",
        "Площадка_mapped",
    ]
    df1[sumcol] = df1[sumcol].astype(float)
    df2 = df1[df1[sumcol] > 0]
    df2 = df2[cols + [sumcol]]
    df3 = df2.pivot_table(
        columns=cols[-1],
        index=cols[:-1],
        values=sumcol,
        aggfunc=["sum", "count"],
    )
    df3 = df3.fillna("")
    # df3.columns = df3.columns.reorder_levels([i for i in range(len(
    # cols))]    )
    df3 = df3.reset_index()
    df3 = df3.sort_values(
        ["HC_HT_mapped", "date_mapped"],
        ascending=[False, True],
    )
    if not dont_push:
        l = Load("GC_pivot")
        l.load(GC_t.finalize(df3))
    Extract.write("../exports/GC_pivot.p", df3)


def run_gc(
    from_pickle: bool = False,
    dont_push: bool = False,
    task: int = None,
):

    task_list = [
        dict(task="deals", status="new",),
        dict(task="deals", status="payed",),
    ]
    if task is not None:
        task_list = [task_list[task]]

    Ge = GC_e()
    Gt = GC_t()

    """
        if ele["status"] == "payed":
            Gt.date_cols_mapping = {
                "date_mapped": {
                    True: 12,
                    False: 5,
                    "check_gt": timedelta(
                        days=5, hours=11
                    ),
                    "from": "Дата оплаты",
                }
            }

    """

    for ele in task_list:
        Gl = Load(f"GC_{ele['task']}_{ele['status']}")
        save_postfix = [v for v in ele.values()]
        save_postfix = "_".join(save_postfix)
        ans = Ge.extract(**ele, from_pickle=from_pickle)
        ans = Gt.transform(ans, save_postfix)
        if not dont_push:
            Gl.load(ans)

    make_pivot(dont_push)
