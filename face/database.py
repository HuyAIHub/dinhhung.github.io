import sqlite3 as lite
import sys
import os

path = os.path.dirname(__file__) + "\\data.db"
con = lite.connect(path)

with con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS img")
    cur.execute("CREATE TABLE img(Id INT, Name TEXT, Image JPG)")
    cur.execute("INSERT INTO img VALUES(1,'thuy','thuy.jpg')")
    cur.execute("INSERT INTO img VALUES(2,'minh','minh.jpg')")
    cur.execute("INSERT INTO img VALUES(3,'chi','chi.jpg')")
    cur.execute("INSERT INTO img VALUES(4,'hoaiphuong','hoaiphuong.jpg')")
    cur.execute("INSERT INTO img VALUES(5,'hoaithuy','hoaithuy.jpg')")
    cur.execute("INSERT INTO img VALUES(6,'giang','giang.jpg')")
    cur.execute("INSERT INTO img VALUES(7,'hung','hung.jpg')")
    cur.execute("INSERT INTO img VALUES(8,'hanh','hanh.jpg')")
    cur.execute("INSERT INTO img VALUES(9,'na','na.jpg')")
    cur.execute("INSERT INTO img VALUES(10,'tam','tam.jpg')")

