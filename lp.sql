CREATE DATABASE licese_plate_db;
USE licese_plate_db;

CREATE TABLE licese_plate (
  id INT AUTO_INCREMENT PRIMARY KEY,
   license_plate_number VARCHAR(45) NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path VARCHAR(255)
);
