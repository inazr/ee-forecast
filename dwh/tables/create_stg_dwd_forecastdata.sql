CREATE TABLE stg_dwd.forecastdata (
	forecasttime timestamp NOT NULL,
	time_of_prediction timestamp NOT NULL,
	stationid varchar(5) NOT NULL,
	ff numeric(4,2) NULL
);
CREATE INDEX index_name ON stg_dwd.forecastdata USING btree (forecasttime, time_of_prediction);
