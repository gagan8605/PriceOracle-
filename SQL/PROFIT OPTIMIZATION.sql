--Top profitable products
SELECT product_id, SUM(profit) AS total_profit
FROM sales_data
GROUP BY product_id
ORDER BY total_profit DESC;

-- Profit by price range
SELECT 
    CASE 
        WHEN price < 200 THEN 'Low'
        WHEN price BETWEEN 200 AND 250 THEN 'Medium'
        ELSE 'High'
    END AS price_range,
    AVG(profit) AS avg_profit
FROM sales_data
GROUP BY price_range;

-- "Medium"	217.25301204819277
-- "Low"	179.33333333333334
-- "High"	594.2069568887443


-- Loss-making transactions
SELECT *
FROM sales_data
WHERE profit < 0;