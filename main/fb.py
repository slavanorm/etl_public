import pandas as pd
from pandas import DataFrame as d
from datetime import datetime, timedelta
from tenacity import retry, wait, wait_fixed
from main.base import (
    Extract,
    Transform,
    Load,
    pretty,
    populate_kwargs,
    today,
    tg_dec,
    for_all_methods,
)
from time import sleep
import requests
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.user import User
import json


@for_all_methods(tg_dec)
class FB_e(Extract):
    year_start = today[:4] + "-01-01"
    year_start = datetime.strptime(
        year_start, "%Y-%m-%d"
    )
    year_n_week_start = year_start + timedelta(
        days=((7 - year_start.weekday() + 4) % 7)
    )
    year_start = year_start.strftime("%Y-%m-%d")
    year_n_week_start = year_n_week_start.strftime(
        "%Y-%m-%d"
    )

    def __init__(self):
        self.accs_fields = ["name", "id", "timezone_id"]
        self.camps_fields = [
            "name",
            "account_id",
            "objective",
        ]

        self.insights_fields = [
            "impressions",
            "reach",
            "spend",
            "cpm",
            "clicks",
            "cpc",
            "ctr",
            "actions",
            "cost_per_action_type",
        ]

        self.insights_params = dict(
            time_increment=7,
            time_range=dict(
                since=self.year_n_week_start,
                until=today,
            ),
        )

        self.access_token = json.load(
            open("../access/facebook.json", "r")
        )
        FacebookAdsApi.init(
            access_token=self.access_token
        )

    def extract(self, short=False):
        # todo adset and ad

        me = User("me")
        accounts = list(
            me.get_ad_accounts(self.accs_fields)
        )
        if short:
            accounts = [accounts[2]]
            self.insights_params["time_range"][
                "since"
            ] = "2020-07-17"
            self.insights_params["time_range"][
                "until"
            ] = "2020-07-23"
        campaigns = []
        for ele in accounts:
            campaigns.extend(
                list(
                    ele.get_campaigns(self.camps_fields)
                )
            )

        insights = {}
        for e in campaigns:
            insights[e["id"]] = list(
                e.get_insights(
                    self.insights_fields,
                    self.insights_params,
                    is_async=False,
                )
            )

        if self.year_start != self.year_n_week_start:
            params2 = dict(
                time_increment=7,
                time_range=dict(
                    since=self.year_start,
                    until=self.year_n_week_start,
                ),
            )
            for e in campaigns:
                insights[e["id"]] += list(
                    e.get_insights(
                        self.insights_fields,
                        params2,
                        is_async=False,
                    )
                )

        accounts = {
            ele["id"][4:]: ele["name"]
            for ele in accounts
        }
        campaigns = {
            ele["id"]: dict(ele) for ele in campaigns
        }

        ans = [
            dict(
                **campaigns[k],
                account_name=accounts[
                    campaigns[k]["account_id"]
                ],
                data=[dict(ele) for ele in v],
            )
            for k, v in insights.items()
            if len(v) > 0
        ]

        Extract.write("../exports/FB_extracted.p", ans)
        return ans


class FB_t(Transform):
    cols_naming = [
        "id",
        "name",
        "date_mapped",
        "account_name",
        "account_id",
        "spend",
        "impressions",
        "reach",
        "cpm",
        "clicks",
        "ctr",
        "objective",
        "actions",
        "cost_per_action_type",
        "date_start",
        "date_stop",
    ]
    date_in_format = "%Y-%m-%d"
    date_cols_mapping = {
        "date_mapped": {
            **{
                timedelta(days=ele): 5
                for ele in range(7)
            },
            "from": "date_stop",
        }
    }
    """
    cols_naming = dict(
        date_start="Дата начала",
        date_stop="Дата конца",
        adaccount="",
        date_mapped="",
        name="Название",
        id="campaign_id",
        impressions="Показ рекламы",
        objective="",
        actions="Результаты",
        cost_per_action_type="Цена за результаты",
        spend="Сумма затрат (RUB)",
        clicks="Клики",
        cpm="CPM, цена 1000 показов",
        cpc="CPC, цена клика по ссылке",
        ctr="CTR, Кликабельность",
    )"""

    def change(self, data):
        data = [
            {**e1, **e}
            for e in data
            for e1 in e["data"]
        ]
        actions_naming = {
            "CONVERSIONS": "onsite_conversion.post_save",
            "LINK_CLICKS": "link_click",
        }

        data = [
            {
                **e,
                "objective": actions_naming[
                    e["objective"]
                ],
            }
            if e["objective"] in actions_naming
            else e
            for e in data
        ]

        filter_fields = [
            "actions",
            "cost_per_action_type",
        ]
        [
            e.update(
                {
                    e2: e1["value"]
                    for e1 in e[e2]
                    if e1["action_type"]
                    == e["objective"].lower()
                }
            )
            for e in data
            for e2 in filter_fields
            if e2 in e
        ]

        data = pd.DataFrame(data).fillna("")

        for ele in ["spend", "reach"]:
            data[ele] = data[ele].astype(float)

        data.loc[
            data["objective"] == "REACH",
            "cost_per_action_type",
        ] = round(data["spend"] / data["reach"], 6)
        data.loc[
            data["objective"] == "REACH", "actions",
        ] = data["reach"]

        data = data.drop(columns="data")

        # data = data.sort_values(by=["date_stop", "id"])
        return data


def run_fb(
    short: bool = False, from_pickle: bool = False
):
    e = FB_e()
    t = FB_t()
    l = Load("FB")
    if from_pickle:
        ans0 = FB_e.read("../exports/FB_extracted.p")
    else:
        ans0 = e.extract(short=short)

    ans1 = t.transform(ans0)
    ans2 = l.load(ans1, freeze=[1, 1])

    v = 1
