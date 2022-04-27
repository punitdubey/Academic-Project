# -*- coding: utf-8 -*-
"""
FILE CONTAINS THE CONSTANT USED IN THE PROJECT CLOSED PRICE PREDICTION

"""
#import
from datetime import datetime

#constants

START_DATE = '2015-08-01'
END_DATE = str(datetime.now().strftime('%Y-%m-%d'))



#symbols of the nifty 50 companies

TICKER = ['ADANIPORTS.NS','ASIANPAINT.NS','AXISBANK.NS','BAJAJ-AUTO.NS',
          'BAJFINANCE.NS','BAJAJFINSV.NS','BPCL.NS','BHARTIARTL.NS','INFRATEL.NS',	
          'BRITANNIA.NS','CIPLA.NS','COALINDIA.NS','DRREDDY.NS',
          'EICHERMOT.NS','GAIL.NS','GRASIM.NS','HCLTECH.NS','HDFC.NS',
          'HDFCBANK.NS','HDFCLIFE.NS','HEROMOTOCO.NS','HINDALCO.NS',
          'HINDUNILVR.NS','ICICIBANK.NS','IOC.NS','INDUSINDBK.NS','INFY.NS',
          'ITC.NS','JSWSTEEL.NS','KOTAKBANK.NS','LT.NS',
          'M&M.NS','MARUTI.NS','NESTLEIND.NS','NTPC.NS','ONGC.NS',
          'POWERGRID.NS','RELIANCE.NS','SHREECEM.NS','SBIN.NS','SUNPHARMA.NS',
          'TCS.NS','TATAMOTORS.NS','TATASTEEL.NS',
          'TECHM.NS','TITAN.NS','ULTRACEMCO.NS','UPL.NS','WIPRO.NS',
          'ZEEL.NS']


#name of the company
COMPANY_NAMES = ['Adani Ports','Asian Paints','Axis Bank','Bajaj Auto',
                 'Bajaj Finance','Bajaj Finserv','Bharat Petroleum',
                 'Bharti Airtel','Bharti Infratel','Britannia Industries',
                 'Cipla','Coal India','Dr. Reddy\'s Laboratories','Eicher Motors',
                 'GAIL','Grasim Industries','HCL Technologies','HDFC','HDFC Bank',
                 'HDFC Life','Hero MotoCorp','Hindalco Industries','Hindustan Unilever',
                 'ICICI Bank','Indian Oil Corporation','IndusInd Bank','Infosys',	
                 'ITC Limited','JSW Steel','Kotak Mahindra Bank','Larsen & Toubro',
                 'Mahindra & Mahindra','Maruti Suzuki','Nestlé India','NTPC',
                 'Oil and Natural Gas Corporation','Power Grid Corporation of India',	
                 'Reliance Industries','Shree Cements','State Bank of India',
                 'Sun Pharmaceutical','Tata Consultancy Services','Tata Motors',
                 'Tata Steel','Tech Mahindra','Titan Company','UltraTech Cement',
                 'United Phosphorus Limited','Wipro','Zee Entertainment Enterprises']


if __name__ == "__main__":
    print("No of Companies : ",len(TICKER))
    print("no of names of the company: ",len(COMPANY_NAMES),"\n\n")
    print("{:42} {:3}\n".format("COMPANY_NAME","SYMBOL"))
    for i in range(len(TICKER)):
        print("{} {:42} : {:3}".format(i+1,COMPANY_NAMES[i],TICKER[i]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    