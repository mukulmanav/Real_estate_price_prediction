Assume that you are a data scientist working for a real estate company such as Zillow.com in the U.S. or MagicBricks.com in India.
Your business manager comes to you and asks you to build a model that can predict the property price based on certain features such as square feet, bedroom, bathroom, location, etc. 




Problem:-

-Check above data points. We have 6 bhk apartment with 1020 sqft. Another one is 8 bhk and total sqft is 600. These are clear data errors that can be removed safely

-remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

-remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

-if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed

-Use One Hot Encoding For Location



Queries:-
    Data cleaning
    Featuring Engineering
    One Hot coding
    Outlier/Anamoly f Detection
    Gridsearch CV


Resources:-
    https://www.youtube.com/watch?v=rdfbcdP75KI&list=PLeo1K3hjS3uu7clOTtwsp94PcHbzqpAdg
    https://github.com/codebasics/py/tree/master/DataScience/BangloreHomePrices
