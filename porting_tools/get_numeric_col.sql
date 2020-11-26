select col.table_schema,
       col.table_name,
       col.ordinal_position as col_id,
       col.column_name,
       col.data_type,
       col.numeric_precision,
       col.numeric_scale
from information_schema.columns col
join information_schema.tables tab on tab.table_schema = col.table_schema
                                   and tab.table_name = col.table_name
                                   and tab.table_type = 'BASE TABLE'
where col.data_type in ('smallint', 'integer', 'bigint', 
                        'decimal', 'numeric', 'real', 'double precision',
                        'smallserial', 'serial', 'bigserial', 'money')
      and col.table_schema not in ('information_schema', 'pg_catalog')
order by col.table_schema,
         col.table_name,
         col.ordinal_position;