import pandas as pd
from pandas import DataFrame as d
import hashlib
from datetime import datetime, timedelta
from main.base import (
    Extract,
    Transform,
    Load,
    pretty,
    today,
)
import json


class SE_e(Extract):
    secret_key = json.load(
        open("../access/senler.json", "r")
    )

    def __init__(self):
        super().__init__(
            url="https://senler.ru/api/",
            v="1.0",
            vk_group_id="120358198",
        )

    def _calculate_hash(self, data, sign=""):
        str_data = []
        for value in data.values():
            if isinstance(value, (tuple, list)):
                str_data += sign.join(value)
            else:
                str_data.append(str(value))
        str_data = sign.join(str_data)
        str_data += self.secret_key
        return hashlib.md5(
            str_data.encode("utf-8")
        ).hexdigest()

    def get(
        self,
        endpoint: str = "subscribers/get",
        data: dict = None,
        **kwargs,
    ):
        if data is None:
            data = {**kwargs, **self.params}
        else:
            data.update({**kwargs, **self.params})
        data["hash"] = self._calculate_hash(data)
        ans = super().extract(
            data=data,
            endpoint=endpoint,
            method="POST",
            all_kw_to_params=False,
        )
        return ans

    def extract(
        self,
        from_pickle=False,
        write: bool = False,
        **kwargs,
    ):
        """
        lets wrap because server answers 1k of rows.
        count:int
        offset:int
        date_first_from:str YYYY-MM-DD HH:MM:SS
        date_first_to:str YYYY-MM-DD HH:MM:SS
        https://help.senler.ru/api/spisok-metodov/podpischiki/poluchenie-podpischikov
        """

        if from_pickle:
            return Extract.read(
                "../exports/SE_extracted.p"
            )

        if "date_first_from" not in kwargs:
            kwargs[
                "date_first_from"
            ] = datetime.today().strftime("%Y-01-01")

        def get_group_names():
            _ = self.get(endpoint="subscriptions/get")
            assert _["success"] == True
            assert len(_["items"]) < 1000
            _ = _["items"]
            _ = {
                int(ele["subscription_id"]): ele["name"]
                for ele in _
                for k, v in ele.items()
            }
            Extract.write(
                "../exports/SE_groups_map.p", _
            )

        get_group_names()
        _ = self.get(**kwargs)
        _ = _["count"]
        rows = int(_)
        retval = []
        for i in range(0, rows, 1000):
            ans = self.get(**kwargs, offset=i,)
            retval.extend(ans["items"])
        if write:
            Extract.write(
                "../exports/SE_extracted.p", retval
            )
        return retval


class SE_t(Transform):
    date_in_format = "%d.%m.%Y"
    cols_naming = dict(
        vk_user_id="ID пользователя ВК",
        utm_source="",
        utm_medium="",
        utm_campaign="",
        utm_content="",
        subscription_id="ID группы",
        sub_name="Название",
        source="Источник",
        date="Дата подписки",
        date_mapped="",
        vk_mapped="",
    )
    date_cols_mapping = {
        "date_mapped": {
            **{
                timedelta(days=ele): 12
                if ele > 3
                else 5
                for ele in range(7)
            },
            "from": "date",
        }
    }
    cols_mapping = [
        dict(
            col="vk_mapped",
            check=["eq", "sub_name_-3:", "reg",],
            result="vk",
        )
    ]

    def change(self, data: list):

        grp_names = Extract.read(
            "../exports/SE_groups_map.p"
        )

        retval = [
            {
                **e1,
                **{
                    k: v
                    for k, v in e.items()
                    if k in self.cols_naming
                },
            }
            for e in data
            for e1 in e["subscriptions"]
            if len(e["subscriptions"]) > 0
        ]

        retval = pd.DataFrame.from_dict(retval)

        retval["sub_name"] = retval[
            "subscription_id"
        ].apply(lambda x: grp_names[x])
        retval["sub_name_-3:"] = retval["sub_name"].str[
            -3:
        ]

        retval.date = retval.date.str[:10]

        return retval


def run_se(from_pickle=False):

    Se = SE_e()
    St = SE_t()
    Sl = Load("senler")
    ans = Se.extract(from_pickle=from_pickle)

    ans = St.transform(ans)
    Sl.load(ans)
