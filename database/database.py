from ..Settings import (
    BASE_DIR, BP_COLS, BP_DATA, ALL_PLORTS,
    sl, os, Literal, traceback, json, logging
    )



DBF = BASE_DIR / 'research.db'

logger = logging.getLogger('standard')

def create_table_config_file(table_names:list[str]):
    """Generates my custom DB table config file"""
    output_data = {}
    def format_column_data(column_name:str, fk:bool|list[str]=False) -> dict[str, str|bool|list[str]]:
        if 'value' in column_name:
            data_type = 'INT'
        elif 'is' in column_name:
            data_type = 'BOOL'
        else:
            data_type = 'TEXT'
        return {
            "name": column_name,
            "data type": data_type,
            "foreign key": fk
        }
    
    for table_name in table_names:
        if table_name == 'base_stats':
            output_data[table_name] = [format_column_data(names) for names in BP_COLS]
     
    return output_data

DB_TABLE_NAMES = [
    'base_stats'
]
DB_TABLE_CONFIG_DATA = create_table_config_file(DB_TABLE_NAMES)


class Database():
    def __init__(self) -> None:
        self.db_loc = os.path.abspath(DBF)
        self._base_class_parameter_amt:int = len(self.__dict__)

    def _create_connection(self) -> tuple[sl.Connection, sl.Cursor]:
        """ Creates/Returns a connection and cursor """
        conn = sl.connect(self.db_loc, check_same_thread=False)
        curs = conn.cursor()
        return conn, curs
    
    def _close_commit(self, connection:sl.Connection) -> None:
        """ Commits changes then closes connection"""
        connection.commit()
        connection.close()
        return

    def create_tables(self) -> bool:
        """ Creates Specimen and Slide tables """
        conn, curs = self._create_connection()

        def create_column_script(col_data:list[dict[str, str|bool]]) -> str:
            """ Concatenates column specifiers; only one column apart of entire execute statement """
            try:
                cols_script:list[str] = [f'{col_specifiers['name']} {col_specifiers['data type'].upper()}' for col_specifiers in col_data]
                fks:list = [f'FOREIGN KEY ({data['name']}) REFERENCES {data['foreign key'][0]}({data['foreign key'][1]})' for data in col_data if isinstance(data['foreign key'], list)]

                # PK must be first in list
                pk = [f'PRIMARY KEY ({col_data[0]['name']})']

                if len(fks) > 1:
                    return (', ').join(cols_script + pk + fks)
                return (', ').join(cols_script + pk)
            except Exception as e:
                traceback.print_exc()
                logger.error(f'Error creating column script for DB table: {col_data}')
        
        # TODO: reference table creation config variables come from CSV
        try:
            for table_name in DB_TABLE_CONFIG_DATA:
                col_script = create_column_script(DB_TABLE_CONFIG_DATA[table_name])
                curs.execute(f'CREATE TABLE IF NOT EXISTS {table_name}({col_script})')
            # curs.execute('ALTER TABLE all_slides ADD CONSTRAINT specimen_id_fk FOREIGN KEY (specimen_aperio_id) REFERENCES specimens(aperio_id)')
            self._close_commit(conn)
            return True
        except sl.OperationalError as e:
            # traceback.print_exc()
            logger.error(f'Error creating table: {table_name}\n{col_script = }\n{e}\n')
            return False

    def _recreate_tables(self):
        conn, curs = self._create_connection()
        tables = [info[0] for info in curs.execute('SELECT name FROM sqlite_master WHERE type=?', ('table',))]
        for table in tables:
            curs.execute(f'DROP TABLE {table}')
        self._close_commit(conn)
        return self.create_tables()
        
    def _parameter_data(self, table:str) -> list[str]:
        """ For DB inserts or user interest """
        return [specifier['name'] for specifier in DB_TABLE_CONFIG_DATA[table]]

    def _parameter_placeholders(self, parameters: list[str]) -> str:
        """ DB insert placeholders """
        placeholders = '?,' * len(parameters)
        return placeholders[:-1]


    def insert_data(self, database_type:Literal['<database name> database'], table_name:str, insert_data:tuple|list) -> bool:
        """Offers alternative to simply insert data if easier"""
        conn, curs = self._create_connection()
        # logger.debug(f'{self.db_loc = }')
        try:
            cols = self._parameter_data(database_type, table_name)
            plch = self._parameter_placeholders(cols)
            # print(f'QUERY: INSERT INTO {self.table_name}{cols} VALUES ({plch})', tuple(insert_data))

            curs.execute(f'INSERT INTO {table_name}({(','.join(cols))}) VALUES ({plch})', tuple(insert_data))
            return True
        except Exception as e:
            logger.error(
                f'\nerror inserting data: {e}\n'
                f'{table_name = } {cols = } {plch = }\n'
                f'{insert_data = }\n'
                )
        finally:
            self._close_commit(conn)

    def update_value(self, table_name:str, set:list[str|int], where:list[str|int]):
        conn, curs = self._create_connection()
        try:
            curs.execute(f'UPDATE {table_name} SET {set[0]}=? WHERE {where[0]}=?', (set[1],where[1]))
        except Exception as e:
            logger.error(
                f'\nerror inserting data: {e}\n'
                f'{table_name = } {set = } {where = }\n'
            )
        finally:
            self._close_commit(conn)
        
    @staticmethod
    def _generate_where_stmt(where_filter:dict[Literal['<database column>'], str|bool|int]) -> dict[Literal['stmt', 'values'], str|list[str]]:
        return {
            'stmt': '=? AND '.join([col for col in where_filter])+'=?',
            'values': [where_filter[col] for col in where_filter]
        }
    

    def db_insert(self) -> tuple[str | bool | int]:
        """ formatted properties to insert into DB """
        # return f'{self._base_class_parameter_amt = }'
        package:list[str | bool | int] = [self.__dict__[parameter] for parameter in self.__dict__][self._base_class_parameter_amt+2:]
        return tuple(package) 

    # def update_single_value(self, set_data:list[Literal['<set_col>, <set_value>']], where_data:list[Literal['<where_col>, <where_value>']]):
    #     if self.__class__ == Specimen:
    #         table = self.specimen_table
    #     elif self.__class__ == WholeSlideImage:
    #         table = self.slide_table
        
    #     conn, curs = self._create_connection()
    #     try:
    #         curs.execute(f'UPDATE {table} SET {set_data[0]}=? WHERE {where_data[0]}=?', (set_data[1], where_data[1]))
    #     except Exception:
    #         logger.error(f'Error updating {table} with data: {set_data = } | {where_data = }')
    #     finally:
    #         self._close_commit(conn)


    def get_property(self, property:Literal['<database column>'], where_row:Literal['<database column>']=None, where_value:str|bool|int=None, count:bool=False, table_name:Literal['all_slides', 'specimens']=None):
        """Search for property from all specimens or specify a specific row"""
        table = self._set_table(table_name)
        conn, curs = self._create_connection()

        try:
            if where_row:
                data = [info[0] for info in curs.execute(f'SELECT {property} FROM {table} WHERE {where_row}=?', (where_value,))]
            else:
                data = [info[0] for info in curs.execute(f'SELECT {property} FROM {table}')]
            
            if count:
                return len(data)
            return data
        except Exception as e:
            print(e)
        finally:
            self._close_commit(conn)
        
        

    def get_multiple_properties(self, count:bool=False, table_name:Literal['all_slides', 'specimens']=None, *properties:list[str], **where_specifiers) -> list[tuple] | int:
        table = self._set_table(table_name)
        conn, curs = self._create_connection()

        data = [info for info in curs.execute(f'SELECT {','.join(properties)} FROM specimens')]
        if where_specifiers:
            where_data = self._generate_where_stmt(where_specifiers)

            data = [info for info in curs.execute(f'SELECT {','.join(properties)} FROM specimens WHERE {where_data['stmt']}', tuple(where_data['values']))]

        self._close_commit(conn)

        if count:
            return len(data)
        return data