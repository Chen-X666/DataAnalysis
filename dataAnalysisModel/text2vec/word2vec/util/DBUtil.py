# -*- coding:utf-8 -*-

from dataAnalysisModel.text2vec.word2vec.util.PropertiesUtil import prop
from dbutils.pooled_db import PooledDB
import importlib



class DbPoolUtil(object):
    def __init__(self, config_file=r'C:/Users/Chen/Desktop/dataAnalysisPlatform/dataAnalysisModel/text2vec/word2vec/config/jdbc.properties', db_type='mysql'):
        """
        初始化
        :param config_file:  配置文件地址
        :param db_type: 数据库类型,支持 mysql, oracle, mssql, sqlite3, hbase
        """
        properties_dic = prop.get_config_dict(config_file)
        self.__db_type = db_type
        if self.__db_type == "mysql":
            config = {
                'host': properties_dic['host_mysql'],
                'port': int(properties_dic['port_mysql']),
                'database': properties_dic['database_mysql'],
                'user': properties_dic['user_mysql'],
                'password': properties_dic['password_mysql'],
                'charset': properties_dic['charset_mysql']
            }
            db_creator = importlib.import_module("pymysql")
            self.__pool = PooledDB(db_creator, maxcached=50, maxconnections=1000, maxusage=1000, **config)
        elif self.__db_type == "oracle":
            config = {
                'user': properties_dic['user_orc'],
                'password': properties_dic['password_orc'],
                'dsn': "/".join(
                    [":".join([properties_dic['host_orc'], properties_dic['port_orc']]),
                     properties_dic['database_orc']]),
                'nencoding': properties_dic['nencoding_orc']
            }
            db_creator = importlib.import_module("cx_Oracle")
            self.__pool = PooledDB(db_creator, maxcached=50, maxconnections=1000, maxusage=1000, **config)
        elif self.__db_type == "mssql":
            config = {
                'host': properties_dic['host_ms'],
                'database': properties_dic['database_ms'],
                'user': properties_dic['user_ms'],
                'password': properties_dic['password_ms'],
                'charset': properties_dic['charset_ms']
            }
            db_creator = importlib.import_module("pymssql")
            self.__pool = PooledDB(db_creator, maxcached=50, maxconnections=1000, maxusage=1000, **config)
        elif self.__db_type == "sqlite3":
            config = {
                'database': properties_dic['database_sqlite3']
            }
            db_creator = importlib.import_module("sqlite3")
            self.__pool = PooledDB(db_creator, maxcached=50, maxconnections=1000, maxusage=1000, **config)
        elif self.__db_type == "hbase":
            config = {
                'url': 'http://{0}:{1}/'.format(properties_dic['host_hb'], properties_dic['port_hb'])
            }
            db_creator = importlib.import_module("phoenixdb")
            self.__pool = PooledDB(db_creator, maxcached=50, maxconnections=1000, maxusage=1000, **config)
        else:
            raise Exception("unsupported database type " + self.__db_type)

    def execute_query(self, sql, args=()):
        """
        执行查询语句，获取结果
        :param sql:sql语句，注意防注入
        :param args:传入参数
        :return:结果集
        """
        result = ()
        conn = self.__pool.connection()
        cur = conn.cursor()
        try:
            cur.execute(sql, args)
            result = cur.fetchall()
        except Exception as e:
            print('异常信息:' + str(e))
        cur.close()
        conn.close()
        return result

    def execute_query_single(self, sql, args=()):
        """
        执行查询语句，获取单行结果
        :param sql:sql语句，注意防注入
        :param args:传入参数
        :return:结果集
        """
        result = ()
        conn = self.__pool.connection()
        cur = conn.cursor()
        try:
            cur.execute(sql, args)
            result = cur.fetchone()
        except Exception as e:
            print('异常信息:' + str(e))
        cur.close()
        conn.close()
        return result

    def execute_iud(self, sql, args=()):
        """
        执行增删改语句
        :param sql:sql语句，注意防注入
        :param args:传入参数
        :return:影响行数,mysql和sqlite有返回值
        """
        conn = self.__pool.connection()
        cur = conn.cursor()
        count = 0
        try:
            result = cur.execute(sql, args)
            conn.commit()
            if self.__db_type == "mysql":
                count = result
            if self.__db_type == "sqlite3":
                count = result.rowcount
        except Exception as e:
            print('异常信息:' + str(e))
            conn.rollback()
        cur.close()
        conn.close()
        return count

    def execute_many_iud(self, sql, args):
        """
        批量执行增删改语句
        :param sql:sql语句，注意防注入
        :param args:参数,内部元组或列表大小与sql语句中参数数量一致
        :return:影响行数，mysql和sqlite有返回值
        """
        conn = self.__pool.connection()
        cur = conn.cursor()
        count = 0
        try:
            result = cur.executemany(sql, args)
            conn.commit()
            if self.__db_type == "mysql":
                count = result
            if self.__db_type == "sqlite3":
                count = result.rowcount
        except Exception as e:
            print('异常信息:' + str(e))
            conn.rollback()
        cur.close()
        conn.close()
        return count

    def execute_proc(self, proc_name, args=()):
        """
        执行存储过程，mysql适用
        :param proc_name:存储过程/函数名
        :param args:参数
        :return:result为结果集，args_out为参数最终结果（用于out，顺序与传参一致）
        """
        result = ()
        args_out = ()
        conn = self.__pool.connection()
        cur = conn.cursor()
        try:
            cur.callproc(proc_name, args)
            result = cur.fetchall()
            if args:
                sql = "select " + ",".join(["_".join(["@", proc_name, str(index)]) for index in range(len(args))])
                cur.execute(sql)
                args_out = cur.fetchone()
            conn.commit()
        except Exception as e:
            print('异常信息:' + str(e))
            conn.rollback()
        cur.close()
        conn.close()
        return result, args_out

    def loop_row(self, obj, fun_name, sql, args=()):
        """
        执行查询语句，并且对游标每行结果反射调用某个处理方法
        主要是考虑一些表记录太大时，不能一次性取出,游标式取数据
        :param obj: 对象或者模块
        :param fun_name:调用方法名
        :param sql:sql语句，注意防注入
        :param args:传入参数
        :return:
        """
        conn = self.__pool.connection()
        cur = conn.cursor()
        try:
            cur.execute(sql, args)
            fun = getattr(obj, fun_name)
            while True:
                row = cur.fetchone()
                if row is None:
                    break
                text = [x if x else "" for x in row]
                words = fun("".join(text))
                yield words
        except Exception as e:
            print('异常信息:' + str(e))
        cur.close()
        conn.close()

    def loop_row_custom(self, sql, args=()):
        """
        执行查询语句，并且对游标每行结果执行某些操作
        主要是考虑一些表记录太大时，不能一次性取出,游标式取数据
        :param sql:sql语句，注意防注入
        :param args:传入参数
        :return:
        """
        conn = self.__pool.connection()
        cur = conn.cursor()
        try:
            cur.execute(sql, args)
            while True:
                row = cur.fetchone()
                if row is None:
                    break
                # 在此编写你想做的操作
        except Exception as e:
            print('异常信息:' + str(e))
        cur.close()
        conn.close()

# if __name__ == "__main__":
#     # 使用demo，工作目录在项目目录的前提下,使用表为TEST2表
#     # from util.DBUtil import dbpool_util
#     # sql1 = """DELETE FROM TEST2"""
#     # result1 = dbpool_util.execute_iud(sql1)
#     # print(result1)
#     #
#     # # mysql和mssql语句的参数使用%s作为占位符，oracle和sqlite使用:数字作为占位符(sqllite还可以用?作为占位符)
#     # sql2 = """INSERT INTO TEST2(id,name) VALUES (%s,%s)"""
#     # # sql2 = """INSERT INTO TEST2(id,name) VALUES (:1,:2)"""
#     # test_args2 = [(1, '王能'), (2, '葬爱'), (3, 'shao'), (5, 'nian'), (8, 'wang')]
#     # result2 = dbpool_util.execute_many_iud(sql2, test_args2)
#     # print(result2)
#     #
#     # sql3 = """SELECT id,name FROM TEST2 """
#     # result3 = dbpool_util.execute_query(sql3)
#     # print(result3)
#     # result3 = dbpool_util.execute_query_single(sql3)
#     # print(result3)
#     # dbpool_util.loop_row_custom(sql3)
#     #
#     # from util.ClassTest import ClsTest
#     #
#     # cla_test = ClsTest()
#     # dbpool_util.loop_row(cla_test, "print_row", sql3)
#     #
#     # import util.ModuleTest as mod_test
#     #
#     # dbpool_util.loop_row(mod_test, "print_row", sql3)
#     # sql4 = """SELECT id,name FROM TEST2 where id = %s"""
#     # # sql4 = """SELECT id,name FROM TEST2 where id = :1"""
#     # test_args4 = (3,)
#     # result4 = dbpool_util.execute_query(sql4, test_args4)
#     # print(result4)
#
#     # 存储过程的验证目前仅在mysql上实现
#     # test_args5 = ('zhou', 'li', '@out1', '@out2')
#     # result5, result6 = dbpool_util.execute_proc("testpro", test_args5)
#     # print(result5)
#     # print(result6)
#
#     # hbase test
#     sql = "SELECT case_serial, CASE_TITLE, CASE_CONTENT, CASE_DATE FROM CASE_INFO limit 50"
#     dbpool_util.execute_query(sql)
