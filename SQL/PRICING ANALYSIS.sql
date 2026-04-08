-- Price vs Profit
SELECT price, AVG(profit) AS avg_profit
FROM sales_data
GROUP BY price
ORDER BY price;

-- Price vs Revenue
SELECT price, AVG(revenue) AS avg_revenue
FROM sales_data
GROUP BY price
ORDER BY price;

-- Price vs Competitor Comparison  
SELECT 
    price, 
    AVG(competitor_price) AS avg_competitor_price,
    AVG(price_diff) AS avg_price_diff
FROM sales_data
GROUP BY price;