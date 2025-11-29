--  CONSULTA 1: ¿ Cuál es el volumen total de ventas y volumen por región?

-- Volument total 

SELECT SUM(Total_Sales) AS Total_Sales
FROM ventas_bmw;

-- Volumen por región

SELECT Region, SUM(Total_Sales) AS Sales_by_Region
FROM ventas_bmw
GROUP BY Region
ORDER BY Sales_by_Region DESC;

-- CONSULTA 2: ¿Cuál es el mix de ventas por modelo y tipo de combustible?

SELECT Model, Fuel_Type,
       SUM(Sales_Volume) AS Total_Volume_Sales,
       SUM(Total_Sales) AS Total_Sales
FROM ventas_bmw
GROUP BY Model, Fuel_Type
ORDER BY Model DESC;

-- CONSULTA 3: ¿ Cuál es la evolución anual de las ventas de cada modelo? Detección de modelos débiles y fuertes.

SELECT Model, Year, 
SUM(Sales_Volume) AS Total_Volume_Sales,
SUM(Total_Sales) AS Total_Sales
FROM ventas_bmw
GROUP BY Model, Year
ORDER BY Model, Year ASC; 

--CONSULTA 4: ¿Cómo son las preferencias de colores por región?

SELECT Region, Color,
SUM(Sales_Volume) AS Total_Volume_Sales,
SUM(Total_Sales) AS Total_Sales
FROM ventas_bmw
GROUP BY Region, Color
ORDER BY Region, Total_Volume_Sales ASC; 

-- CONSULTA 5: ¿Cuál es el modelo más vendido por región?

WITH RegionSales AS (
    SELECT 
        Region, 
        Model, 
        SUM(Sales_Volume) AS Total_Volume_Sales,
        SUM(Total_Sales) AS Total_Sales,        
        RANK() OVER (PARTITION BY Region ORDER BY SUM(Sales_Volume) DESC) as rnk
    FROM ventas_bmw
    GROUP BY Region, Model
)
SELECT Region, Model, Total_Volume_Sales, Total_Sales
FROM RegionSales
WHERE rnk = 1;

-- CONSULTA 6: ¿Cuál es la participación de cada modelo en el total de ventas?

SELECT Model,
       SUM(Total_Sales) AS Total_Sales,
       SUM(Total_Sales) * 100.0 / (SELECT SUM(Total_Sales) FROM ventas_bmw) AS Market_Share
FROM ventas_bmw
GROUP BY Model
ORDER BY Total_Sales DESC;

-- CONSULTA 7: ¿Cuál es el Kilometraje promedio por tipo de combustible?

SELECT Fuel_Type,
       AVG(Mileage_KM) AS Avg_Mileage
FROM ventas_bmw
GROUP BY Fuel_Type
ORDER BY Avg_Mileage DESC;

-- CONSULTA 8: ¿Cuál es el total de ventas por tipo de transmisión?

SELECT Transmission,
       SUM(Sales_Volume) AS Total_Volume_Sales,
       SUM(Total_Sales) AS Total_Sales
FROM ventas_bmw
GROUP BY Transmission
ORDER BY Total_Sales ASC;

-- CONSULTA 9: ¿Cómo ha evolucionado la venta de los modelos manuales y automáticos en el tiempo?

SELECT Transmission, Year, 
SUM(Sales_Volume) AS Total_Volume_Sales,
SUM(Total_Sales) AS Total_Sales
FROM ventas_bmw
GROUP BY Transmission, Year
ORDER BY Transmission, Year ASC; 

-- CONSULTA 10: ¿ Cuál es el rendimiento de modelos Premium vs Gama media?

SELECT Category,
       SUM(Sales_Volume) AS Total_Volume_Sales,
       SUM(Total_Sales) AS Total_Sales
FROM (
    SELECT *,
        CASE 
            WHEN Model IN ('i8', 'X6', 'X5', 'M5', 'M3', '7 Series', '5 Series') THEN 'Premium'
            WHEN Model IN ('i3', 'X3', 'X1', '3 Series') THEN 'Gama_media'
            ELSE 'Other'
        END AS Category
    FROM ventas_bmw
)
WHERE Category IN ('Premium', 'Gama_media')
GROUP BY Category;


