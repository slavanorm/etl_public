import pandas as pd
from functools import reduce
from main.base import (
    Extract,
    pretty,
    Load,
    Transform,
)
import gspread as gs
from gspread import utils as gu


def prepare_data():

    work = [
        "VK",
        "FB",
        "SE",
        "GC",
    ]
    params = {}

    for ele in work:
        if ele == "GC":
            ele += "_pivot"
        params[ele] = Extract.read(
            f"../exports/{ele}.p"
        )

    def vk_p(df1):
        sumcol = ["Потрачено"]
        cols = ["date_mapped"]
        df1[sumcol] = df1[sumcol].astype(float)
        df2 = df1.pivot_table(
            index=cols, values=sumcol, aggfunc="sum",
        )
        df2["Потрачено с учетом бонуса"] = (
            0.7 * df2["Потрачено"]
        )
        prefix = "vk_"
        c = list(df2.columns)
        c = [
            (prefix + e)
            for e in c
            if e != "date_mapped"
        ]
        df2.columns = c
        return df2

    def fb_p(df1):
        sumcol = ["spend"]
        cols = ["date_mapped"]
        df1[sumcol] = df1[sumcol].astype(float)
        df2 = df1.pivot_table(
            index=cols, values=sumcol, aggfunc="sum",
        )

        prefix = "fb_"
        c = list(df2.columns)
        c = [
            (prefix + e)
            for e in c
            if e != "date_mapped"
        ]
        df2.columns = c
        return df2

    def se_p(df1):
        sumcol = ["vk_mapped"][0]
        cols = ["date_mapped"][0]
        df1 = df1[df1[sumcol] == "vk"][[sumcol, cols]]
        df2 = df1.groupby(cols).count().reset_index()

        df2.columns = [
            "date_mapped",
            "se_users_reg_count",
        ]

        return df2

    def gc_p(df1):
        df2 = df1.set_index(
            ["date_mapped", "HC_HT_mapped"]
        )
        df2 = df2.unstack()
        df2.columns = df2.columns.reorder_levels(
            [1, 2, 0]
        )
        df2 = df2.reset_index()
        return df2

    ans = [
        fb_p(params["FB"]),
        se_p(params["SE"]),
        vk_p(params["VK"]),
        gc_p(params["GC_pivot"]),
    ]

    ans = reduce(
        lambda x, y: pd.merge(
            x, y, on="date_mapped", how="outer"
        ),
        ans,
    )

    ans = ans.fillna("")
    Extract.write("../exports/final_pivot.p", ans)
    return ans


def col_to_a1(col):
    MAGIC_NUMBER = 64
    div = col
    column_label = ""

    while div:
        (div, mod) = divmod(div, 26)
        if mod == 0:
            mod = 26
            div -= 1
        column_label = (
            chr(mod + MAGIC_NUMBER) + column_label
        )
    return column_label


def fill_formulas():
    def fill_down_base(
        worksheet, col: (str, int), cell_value: str,
    ):
        """
        create ranges from cols
        get table length for rows


        """
        if isinstance(col, int):
            col = col_to_a1(col)
        col = col.upper()
        range1 = gu.a1_range_to_grid_range(
            f"{col}{minrow}:{col}{maxrow}",
            sheet_id=worksheet.id,
        )
        return dict(rng=range1, val=cell_value)

    def fill_down(spreadsheet, reqs: list):
        reqs2 = dict(
            requests=[
                dict(
                    repeatCell=dict(
                        range=e["rng"],
                        fields="*",
                        cell=dict(
                            userEnteredValue=dict(
                                formulaValue=e["val"]
                            )
                        ),
                    )
                )
                for e in reqs
            ]
        )
        spreadsheet.batch_update(reqs2)

    req = [
        fill_down_base(wks, i, ele)
        for i, ele in enumerate(first_row, 1)
        if "=" in ele
    ]
    fill_down(spr, req)


def write_data():
    cols_mapping = {
        "date_mapped": "Дата интенсива",
        "vk_Потрачено": "Потрачено, руб.",
        "fb_spend": "Сумма затрат (RUB)",
        "se_users_reg_count": "Вконтакте /рег",
        ("vk", "HC", "sum"): "Вконтакте /продаж HC",
        ("ig", "HC", "sum"): "Инстаграм /продаж HC",
        ("google", "HC", "sum"): "Гугл /продаж HC",
        (
            "partner",
            "HC",
            "sum",
        ): "Партнеры /продаж HC",
        ("others", "HC", "sum",): "Другие /продаж HC",
        ("vk", "HT", "sum"): "Вконтакте /продаж HT",
        ("ig", "HT", "sum"): "Инстаграм /продаж HT",
        ("google", "HT", "sum"): "Гугл /продаж HT",
        (
            "partner",
            "HT",
            "sum",
        ): "Партнеры /продаж HT",
        ("others", "HT", "sum"): "Другие /продаж HT",
        ("vk", "HC", "count"): "Вконтакте /счет HC",
        ("ig", "HC", "count"): "Инстаграм /счет HC",
        ("google", "HC", "count"): "Гугл /счет HC",
        (
            "partner",
            "HC",
            "count",
        ): "Партнеры /счет HC",
        ("others", "HC", "count"): "Другие /счет HC",
    }

    cols_index = dict()
    for k, v in cols_mapping.items():
        col = (
            second_row.index(v) + 1
        )  # list starts w 0, gspread with 1
        col = col_to_a1(col)
        cols_index[k] = "".join(
            [col, minrow, ":", col, maxrow]
        )
    for col, address in cols_index.items():
        data = [
            [str(e)]
            for e in pivot[col].to_numpy()
            if col in pivot.columns
        ]
        wks.update(
            address,
            data,
            value_input_option="USER_ENTERED",
        )

    v = 1


def check_maxrow(pivot):
    diff = (
        pivot.shape[0] - 1 - wks.row_count + int(minrow)
    )
    if diff > 0:
        wks.add_rows(diff)
    return str(wks.row_count)


def run_pivot():
    global pivot, first_row, minrow, maxrow, second_row

    pivot = prepare_data()

    first_row = wks.row_values(
        1, value_render_option="FORMULA"
    )
    minrow = "3"
    maxrow = check_maxrow(pivot)
    second_row = wks.row_values(2)

    write_data()
    fill_formulas()

    v = 1


pivot, first_row, minrow, maxrow, second_row = (
    None,
) * 5

spreadsheet_name = "Нейрософия.Выгрузка"
worksheet_name = "сводная"
ga = gs.service_account("../access/google.json")
spr = ga.open(spreadsheet_name)
wks = spr.worksheet(worksheet_name)
