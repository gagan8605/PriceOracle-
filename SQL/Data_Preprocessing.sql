CREATE TABLE sales_data (
    product_id VARCHAR(50),
    quantity INT,
    region VARCHAR(50),
    price FLOAT,
    competitor_price FLOAT,
    cost_price FLOAT,
    is_discount INT,
    day_of_week INT,
    month INT,
    profit FLOAT,
    revenue FLOAT,
    price_diff FLOAT,
    date DATE
);

SELECT * from sales_data;

--Total records
SELECT COUNT(*) AS total_rows FROM sales_data; -- 113815

-- Unique products
SELECT COUNT(DISTINCT product_id) AS total_products FROM sales_data; -- 7124

-- Date range
SELECT MIN(date) AS start_date, MAX(date) AS end_date FROM sales_data; -- "2022-01-01"	"2022-12-31"


-- Total demand per product
SELECT product_id, SUM(quantity) AS total_demand
FROM sales_data
GROUP BY product_id
ORDER BY total_demand DESC;

-- Average demand per price
SELECT price, AVG(quantity) AS avg_demand
FROM sales_data
GROUP BY price
ORDER BY price;

-- Demand trend by month
SELECT month, SUM(quantity) AS total_demand
FROM sales_data
GROUP BY month
ORDER BY month;

-- 1	28671
-- 2	26079
-- 3	29153
-- 4	28129
-- 5	28654
-- 6	27963
-- 7	29052
-- 8	28826
-- 9	28197
-- 10	28861
-- 11	28285
-- 12	28893


