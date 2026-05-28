-- SQL to create `users` table in `lon-default` database
CREATE DATABASE IF NOT EXISTS `lon-default`;
USE `lon-default`;

CREATE TABLE IF NOT EXISTS `users` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(150) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(50) DEFAULT 'user',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Example: to insert a user manually (replace HASHED_PASSWO
-- INSERT INTO `users` (username, password_hash, role) VALUES ('admin', 'HASHED_PASSWORD', 'admin');
