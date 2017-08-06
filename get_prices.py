import pandas as pd

def get_prices(code, start, end=None, interval='d', kind="Open", verbose=True):
    # get stock price
    # input:
    # code,,, stock code
    # start,,, term of get data
    # end,,, term of get data
    # interval,,,
    # kind,,,kind of stock data

    # output;
    # stock data; list

    base = 'http://info.finance.yahoo.co.jp/history/?code={0}.T&{1}&{2}&tm={3}&p={4}'

    try:
        start = pd.to_datetime(start)
    except:
        raise ValueError("start is invalid")

    if end == None:
        end = pd.to_datetime(pd.datetime.now())
    else :
        try:
            end = pd.to_datetime(end)
        except:
            raise ValueError("end is invalid")

    start = 'sy={0}&sm={1}&sd={2}'.format(start.year, start.month, start.day)
    end = 'ey={0}&em={1}&ed={2}'.format(end.year, end.month, end.day)

    if interval not in ['d', 'w', 'm', 'v']:
        raise ValueError("Invalid interval: valid values are 'd', 'w', 'm' and 'v'")

    p = 1
    results = []
    while True:
        url = base.format(code, start, end, interval, p)
        try:
            tables = pd.read_html(url, header=0)
        except:
            raise ValueError("invalid code")

        if len(tables) < 2 or len(tables[1]) == 0: break
        results.append(tables[1])
        p += 1

    try:
        result = pd.concat(results, ignore_index=True)
    except:
        raise ValueError("invalid start of end")

    result.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    if interval == 'm':
        result['Date'] = pd.to_datetime(result['Date'], format='%Y年%m月')
    else:
        result['Date'] = pd.to_datetime(result['Date'], format='%Y年%m月%d日')

    result = result.set_index('Date')
    result = result.sort_index()

    if verbose:
        print(result)

    result = list(result.ix[0:, kind])

    return result

if __name__ == "__main__":
    try:
        price = get_prices(9104, start="2016/05/01", end="2016/07/01", interval="m", kind="Open", verbose=True)
    except:
        print("get price error")
