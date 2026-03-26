import json
import os
import requests
import pandas as pd
import datetime


# 已移除全局 os.environ 代理设置，避免污染全局网络请求


def get_binance_futures_info():
    """
    拉取币安所有U本位合约的当前资金费率及24小时涨跌幅，并保存为CSV
    """
    base_url = "https://fapi.binance.com"
    ticker_endpoint = "/fapi/v1/ticker/24hr"
    funding_endpoint = "/fapi/v1/premiumIndex"

    # 预定义标准的目标空格式：保证在任何失败情况下返回的数据结构一致，调用方不会崩溃
    empty_df = pd.DataFrame(columns=['合约 (Symbol)', '最新价格', '24H涨跌幅 (%)', '当前资金费率 (%)'])

    # 局部代理配置，只针对当前函数生效
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }

    try:
        print("正在从币安拉取数据，请稍候...")

        # 1. 获取24小时行情数据 (注入局部代理)
        ticker_res = requests.get(base_url + ticker_endpoint, proxies=proxies, timeout=10)
        ticker_res.raise_for_status()
        ticker_data = ticker_res.json()

        # 2. 获取溢价指数数据 (资金费率) (注入局部代理)
        funding_res = requests.get(base_url + funding_endpoint, proxies=proxies, timeout=10)
        funding_res.raise_for_status()
        funding_data = funding_res.json()

        # 3. 数据处理与合并
        merged_data = {}
        for item in ticker_data:
            symbol = item['symbol']
            merged_data[symbol] = {
                '合约 (Symbol)': symbol,
                '最新价格': float(item['lastPrice']),
                '24H涨跌幅 (%)': float(item['priceChangePercent'])
            }

        for item in funding_data:
            symbol = item['symbol']
            if symbol in merged_data:
                if 'lastFundingRate' in item and item['lastFundingRate']:
                    funding_rate = float(item['lastFundingRate']) * 100
                else:
                    funding_rate = 0.0
                merged_data[symbol]['当前资金费率 (%)'] = funding_rate

        # 4. 转换为 Pandas DataFrame
        data_list = list(merged_data.values())
        df = pd.DataFrame(data_list)

        # 兜底：如果获取到的数据刚好为空列表，直接返回标准空格式
        if df.empty:
            return empty_df

        # 5. 数据清洗：只保留USDT永续合约并排序
        df = df[df['合约 (Symbol)'].str.endswith('USDT')]
        df = df.sort_values(by='当前资金费率 (%)', ascending=False)
        df = df.reset_index(drop=True)

        return df

    except Exception as e:
        # 捕获所有异常(断网/超时/数据异常)，确保不影响调用方，正常返回空的目标格式
        print(f"处理数据时发生错误或网络异常: {e}")
        return empty_df


def publish_to_binance_square(api_key, text_content):
    """
    根据 Binance Skills Hub 最新规范向币安广场发送发帖请求。

    :param api_key: 币安广场创作者中心生成的 OpenAPI Key
    :param text_content: 帖子正文内容
    """
    # 官方文档指定的 API 端点
    url = "https://www.binance.com/bapi/composite/v1/public/pgc/openApi/content/add"

    # 构造请求头 (Header) - 严格按照规范说明
    headers = {
        "Content-Type": "application/json",
        "clienttype": "binanceSkill",
        "X-Square-OpenAPI-Key": api_key
    }

    # 核心修改：将 "content" 改为官方最新要求的 "bodyTextOnly"
    payload = {
        "bodyTextOnly": text_content
    }

    try:
        # 隐藏掉中间的 API Key 字符，避免在控制台全量打印（官方安全规范）
        masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
        print(f"⏳ 正在向币安广场发送发帖请求 (使用 Key: {masked_key})...")

        response = requests.post(url, headers=headers, json=payload, timeout=15)
        result = response.json()

        # 判断业务是否成功
        if result.get('success') or str(result.get('code')) == '000000':
            print("✅ 恭喜！发布成功！")

            # 根据规范，获取返回数据中的 id 来拼接你的帖子直达链接
            post_id = result.get('data', {}).get('id')
            if post_id:
                post_url = f"https://www.binance.com/square/post/{post_id}"
                print(f"👉 帖子链接: {post_url}")
            return True

        else:
            print("❌ 发布失败，接口返回信息:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

            # 如果报 220009 错误，说明达到了每日发帖上限
            if str(result.get('code')) == '220009':
                print("💡 提示：该错误码通常表示已达到每日发帖数量上限 (Daily post limit)。")
            return False

    except requests.exceptions.RequestException as e:
        print(f"🚨 网络请求发生异常: {e}")
        return False


def get_binance_feed(token="DOGE", desire_count=20, orderBy=2):
    """
    获取币安 Feed 数据
    :param token: 币种名称
    :param desire_count: 期望获取的数据条数，默认为20
    :param orderBy: 1 代表热门 2代表最新
    :return: 目标 vos 列表，失败或无数据时返回 []
    """
    url = "https://www.binance.com/bapi/composite/v4/friendly/pgc/feed/trade/list"

    # 将所有复杂的请求头原样照搬
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9",
        "bnc-location": "",
        "bnc-time-zone": "Asia/Shanghai",
        "bnc-uuid": "e7ea5e07-ca28-4bba-873a-6fd97d181f8b",
        "clienttype": "web",
        "content-type": "application/json",
        "cookie": "aws-waf-token=f51c8e86-3370-4070-ac5c-415f0e361552:AQoAqSoJApETAAAA:MN8Jeh3xuoAr+Cbt162w+olsrObZE8SnSIMlCcLkmP/ameWfIhlg0IspO3dQkPqIgjhIgIPLyuTrt2Xm/fwrARe7fqULwLxvuu1TsSc6gPPzMPxxYVscNKQSvpjcyocs25gs6BLPlRRT//ci+WLICbK9FNdByJeFOnPmEDGjnWNEZLajJGcRx43d/+OTy1MS5lA=; theme=dark; bnc-uuid=e7ea5e07-ca28-4bba-873a-6fd97d181f8b; userPreferredCurrency=USD_USD; sajssdk_2015_cross_new_user=1; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2219d27b9f9131bb2-0eb21c83c1e9f28-26061f51-2359296-19d27b9f91428e6%22%2C%22first_id%22%3A%22%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMTlkMjdiOWY5MTMxYmIyLTBlYjIxYzgzYzFlOWYyOC0yNjA2MWY1MS0yMzU5Mjk2LTE5ZDI3YjlmOTE0MjhlNiJ9%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%22%2C%22value%22%3A%22%22%7D%7D; _gid=GA1.2.1029914182.1774487995; BNC_FV_KEY=3375a92aeff0a1088828bb63852a699067f1724f; BNC_FV_KEY_T=101-2UWasrKs4ekF15fS%2BKlMqv5SebKUF5K00BWqJkLjQ8iE0Ib4Py9K8GwrvWnJnJZI7853k1a1F%2B3g6rYZAsNd%2Fg%3D%3D-LA%2F8O0qqYJjNRqhVsXf%2F2g%3D%3D-d7; BNC_FV_KEY_EXPIRE=1774509598712; changeBasisTimeZone=; _gcl_au=1.1.2053461943.1774488311; g_state={\"i_l\":0,\"i_ll\":1774488312411,\"i_b\":\"Hti4UOgdLZEPsnuPqrAwbnp64On+5RvvhciOCmGyePQ\",\"i_e\":{\"enable_itp_optimization\":0}}; _uetsid=a3965c7028b211f1a285650e7063e82b; _uetvid=a3968b6028b211f1a43b73be74d3c81d; OptanonConsent=isGpcEnabled=0&datestamp=Thu+Mar+26+2026+09%3A26%3A23+GMT%2B0800+(%E4%B8%AD%E5%9B%BD%E6%A0%87%E5%87%86%E6%97%B6%E9%97%B4)&version=202506.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=f791b433-38c4-4a9e-a1a9-f05cb0758f68&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0003%3A1%2CC0004%3A1%2CC0002%3A1&AwaitingReconsent=false; _ga_3WP50LGEEC=GS2.1.s1774487994$o1$g1$t1774488393$j26$l0$h0; _ga=GA1.1.596932932.1774487995",
        "csrftoken": "d41d8cd98f00b204e9800998ecf8427e",
        "device-info": "eyJzY3JlZW5fcmVzb2x1dGlvbiI6IjIwNDgsMTE1MiIsImF2YWlsYWJsZV9zY3JlZW5fcmVzb2x1dGlvbiI6IjIwNDgsMTEwNCIsInN5c3RlbV92ZXJzaW9uIjoiV2luZG93cyAxMCIsImJyYW5kX21vZGVsIjoidW5rbm93biIsInN5c3RlbV9sYW5nIjoiemgtQ04iLCJ0aW1lem9uZSI6IkdNVCswODowMCIsInRpbWV6b25lT2Zmc2V0IjotNDgwLCJ1c2VyX2FnZW50IjoiTW96aWxsYS81LjAgKFdpbmRvd3MgTlQgMTAuMDsgV2luNjQ7IHg2NCkgQXBwbGVXZWJLaXQvNTM3LjM2IChLSFRNTCwgbGlrZSBHZWNrbykgQ2hyb21lLzE0Ni4wLjAuMCBTYWZhcmkvNTM3LjM2IiwibGlzdF9wbHVnaW4iOiJQREYgVmlld2VyLENocm9tZSBQREYgVmlld2VyLENocm9taXVtIFBERiBWaWV3ZXIsTWljcm9zb2Z0IEVkZ2UgUERGIFZpZXdlcixXZWJLaXQgYnVpbHQtaW4gUERGIiwiY2FudmFzX2NvZGUiOiJmZDJkMWY1NyIsIndlYmdsX3ZlbmRvciI6Ikdvb2dsZSBJbmMuIChOVklESUEpIiwid2ViZ2xfcmVuZGVyZXIiOiJBTkdMRSAoTlZJRElBLCBOVklESUEgR2VGb3JjZSBSVFggMzA5MCAoMHgwMDAwMjIwNCkgRGlyZWN0M0QxMSB2c181XzAgcHNfNV8wLCBEM0QxMSkiLCJhdWRpbyI6IjEyNC4wNDM0NzUyNzUxNjA3NCIsInBsYXRmb3JtIjoiV2luMzIiLCJ3ZWJfdGltZXpvbmUiOiJBc2lhL1NoYW5naGFpIiwiZGV2aWNlX25hbWUiOiJDaHJvbWUgVjE0Ni4wLjAuMCAoV2luZG93cykiLCJmaW5nZXJwcmludCI6ImQzM2I2OTcxYTY3NWUxN2RkODJiMGZmOTFkMDcyOTczIiwiZGV2aWNlX2lkIjoiIiwicmVsYXRlZF9kZXZpY2VfaWRzIjoiIn0=",
        "fvideo-id": "3375a92aeff0a1088828bb63852a699067f1724f",
        "fvideo-token": "G1mTiXfnYgwo6jnjqbRxnfp79DA/laP4sr+ns+oaDK9aiFpf+3KBknh1t2NFUX3uDHKhMIIGMkNm6mxVjkc/5emieR7Zh/5bhsg8lDORfAC1ob7S3a3EyVPC18b+NtSRkMe8dwmO42iQ4ub6MQHHS9CF1KQXIkKpbwvkVP1+JQTNmJfeLORXBEsDy9s+ZT+d0=3c",
        "lang": "zh-CN",
        "origin": "https://www.binance.com",
        "referer": "https://www.binance.com/zh-CN/square/community?token=DOGE",
        "sec-ch-ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        "x-trace-id": "417ca1cb-11cb-42dd-a7fa-4bbbcb2efbdb",
        "x-ui-request-trace": "417ca1cb-11cb-42dd-a7fa-4bbbcb2efbdb"
    }

    # 动态构建 Payload，使用 desire_count 控制 pageSize
    payload = {
        "token": token,
        "pageIndex": 1,
        "pageSize": desire_count,
        "scene": 2,
        "orderBy": orderBy,
        "contentIds": []
    }

    # 局部代理配置，只影响当前请求
    proxies = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=10)
        response.raise_for_status()
        res_data = response.json()

        # 安全解析：确保结构存在且类型正确，避免 KeyError 或 NoneType 报错
        if res_data and isinstance(res_data.get("data"), dict):
            vos = res_data["data"].get("vos")
            if isinstance(vos, list):
                return vos

        return []
    except Exception as e:
        # 捕获所有异常(断网、代理失效、JSON解析失败等)，确保不影响调用方
        print(f"获取Feed请求失败: {e}")
        return []


if __name__ == "__main__":

    print("=" * 40)
    print("1. 测试: 获取币安 Feed 数据")
    print("=" * 40)
    feed_data = get_binance_feed(token="DOGE", desire_count=2)
    print(f"✅ 获取到 {len(feed_data)} 条 Feed 数据。\n")

    print("=" * 40)
    print("2. 测试: 获取合约资金费率及涨跌幅")
    print("=" * 40)
    result_df = get_binance_futures_info()

    # 因为做过防崩溃处理，这里可以直接放心使用 result_df 而不怕 NoneType 报错
    if not result_df.empty:
        filename = "binance_futures_data.csv"
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ 成功获取 {len(result_df)} 条合约数据，已保存至: {filename}")
        print(f"预览前3条数据:\n{result_df.head(3)}\n")
    else:
        print("⚠️ 未能获取到合约数据 (返回了空 DataFrame)。\n")

    print("=" * 40)
    print("3. 测试: 广场自动化发帖")
    print("=" * 40)
    YOUR_SQUARE_API_KEY = "替换成你的_API_KEY"
    test_text = "这是一条基于最新 Binance Skills API 规范的自动化发帖测试。🚀 #API测试"

    if YOUR_SQUARE_API_KEY != "替换成你的_API_KEY":
        publish_to_binance_square(api_key=YOUR_SQUARE_API_KEY, text_content=test_text)
    else:
        print("⚠️ 请先在代码中填入你的 API Key 再运行发帖测试！\n")