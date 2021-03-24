import pandas as pd
from webull import paper_webull
wb = paper_webull()
data = wb.login('etlivefree@gmail.com','liveFree0815!','TBpython','422834')
wb.refresh_login()

def execute_webull_option(optionId,limit_price,action='BUY',quantity=1,order_type='LMT',enforce='GTC'):
    # Places an options trade from an option Id via the Webull Platform; returns the result of the trade
    result = wb.place_order_option(optionId=optionId, lmtPrice=limit_price, action=action, orderType=order_type, enforce=enforce,quant=quantity)
    return result

def get_webull_options(ticker):
    options = wb.get_options(ticker)

    options_dict = {}
    official_options_list = []
    #loops for inputting options data in a dataframe
    for option_iter in range(0,len(options)):
        # Loop for determining Strike Price, Calls, Puts
        list_option = options[option_iter]
        #print(list_option)
        for key in list_option:
            #strike_price = list_option[key]
            if key != 'strikePrice':
                temp_dict = list_option[key]
                official_options_list.append(temp_dict)
    #print(official_options_list)
    options_df = pd.DataFrame(official_options_list)
    # Splitting Ask & Bid List Series Data into separate columns
    options_df[['Ask_Price']] = options_df.askList[0][0]['price']
    options_df[['Ask_Volume']] = options_df.askList[0][0]['volume']
    options_df[['Bid_Price']] = options_df.bidList[0][0]['price']
    options_df[['Bid_Volume']] = options_df.bidList[0][0]['volume']
    # Setting the DF index to the unique ticker options id
    options_df.set_index('tickerId', inplace=True)
    #reordering columns
    official_options_df = options_df[['unSymbol','symbol','direction','strikePrice','Ask_Price','Ask_Volume','Bid_Price','Bid_Volume','expireDate','tradeTime','tradeStamp','impVol','volume','close','preClose','open','high','low','delta','vega','gamma','theta','rho','changeRatio','change','weekly','activeLevel','openIntChange']]
    return official_options_df