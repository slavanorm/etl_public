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
    year_start,
)
from time import sleep
import json


class VK_e(Extract):
    def __init__(self):
        self.stats_fields = [
            "clicks",
            "impressions",
            "spent",
            "reach",
            "ctr",
            "eCPC",
            "join_rate",
        ]
        super().__init__(
            url="https://api.vk.com/method/",
            access_token=json.load(
                open("../access/vk.json", "r")
            ),
            v="5.62",
            account_id=1600240232,  # ID AdAccount
            client_id=193920034,  # ID приложения с доступом
            include_deleted=1,
            # include_deleted=0,
        )

    def get(
        self, endpoint: str = "", **kwargs,
    ):
        ans = super().extract(endpoint, **kwargs,)
        try:
            ans = ans["response"]
        except:
            print(ans["error"])
        return ans

    def get_ads(
        self,
        ids: list,
        offset: int = 0,
        recursive=False,
    ):
        # https://vk.com/dev/ads.getAds
        retval = self.get(
            ids=",".join(ids),
            offset=offset * 2000,
            # endpoint="ads.getAds",
            endpoint="ads.getStatistics",
            include_deleted=1,
            ids_type="campaign",
            period="day",
            date_from=year_start,
            date_to=today,
            stats_fields=",".join(self.stats_fields),
        )
        sleep(0.3)
        if recursive and len(retval) == 2000:
            ans2 = self.get_ads(ids, offset + 1,)
            retval.extend(ans2)
        return retval

    def extract(self, full: bool = True):
        ans = self.get("ads.getCampaigns")
        campaigns = pd.json_normalize(ans)
        campaigns_l = campaigns["id"].to_list()
        campaigns_l = [str(e) for e in campaigns_l]

        ans = self.get_ads(ids=campaigns_l)
        Extract.write("../exports/VK_extracted.p", ans)
        return ans


class VK_t(Transform):

    cols_naming = dict(
        day="Дата создания",
        date_mapped="",
        campaign_id="",
        join_rate="Вступления",
        clicks="Переходы",
        eCPC="eCPC,руб",
        spent="Потрачено",
        reach="Охват",
        ctr="CTR,%",
    )
    date_in_format = "%Y-%m-%d"
    date_cols_mapping = {
        "date_mapped": {
            **{
                timedelta(days=ele): 12
                if ele > 3
                else 5
                for ele in range(7)
            },
            "from": "day",
        }
    }

    def change(self, data):
        data = [
            {**e1, "campaign_id": e["id"]}
            for e in data
            for e1 in e["stats"]
        ]
        data = pd.json_normalize(data).fillna(0)

        data.day = data.day.str[:10]

        return data


def run_vk():

    e = VK_e()
    t = VK_t()
    l = Load("VK")
    ans = e.extract()
    ans = t.transform(ans)
    l.load(ans)


v = 1
