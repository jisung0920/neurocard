"""Registry of datasets and schemas."""
import collections
import os
import pickle

import numpy as np
import pandas as pd

import collections
from common import CsvTable


def CachedReadCsv(filepath, **kwargs):
    """Wrapper around pd.read_csv(); accepts same arguments."""
    parsed_path = filepath[:-4] + '.df'
    if os.path.exists(parsed_path):
        with open(parsed_path, 'rb') as f:
            df = pickle.load(f)
        assert isinstance(df, pd.DataFrame), type(df)
        print('Loaded parsed csv from', parsed_path)
    else:
        df = pd.read_csv(filepath, **kwargs)
        with open(parsed_path, 'wb') as f:
            # Use protocol=4 since we expect df >= 4GB.
            pickle.dump(df, f, protocol=4)
        print('Saved parsed csv to', parsed_path)
    return df

class TPC_DS(object) :
    ALIAS_TO_TABLE_NAME = {
        'ss' : 'store_sales',
        'sr' : 'store_returns',
        'cs' : 'catalog_sales',
        'cr' : 'catalog_returns',
        'ws' : 'web_sales',
        'wr' : 'web_returns',
        'inv' : 'inventory',
        's' : 'store',
        'cc' : 'call_center',
        'cp' : 'catalog_page',
        'web' : 'web_site',
        'wp' : 'web_page',
        'w' : 'warehouse',
        'c' : 'customer',
        'ca' : 'catalog_address',
        'cd' : 'customer_demographics',
        'd' : 'date_dim',
        'hd' : 'houshold_demographics',
        'i' : 'item',
        'ib' : 'income_band',
        'p' : 'promotion', 
        'r' : 'reason',
        'sm' : 'ship_mode', 
        't' : 'time_dim',
        # dv is not formal abbreviation in documents
        'dv' : 'dbgen_version', 
        
    }
    CATEGORICAL_COLUMNS = collections.defaultdict(list,
        {
            'call_center': ['cc_call_center_sk',  'cc_call_center_id',  'cc_rec_start_date',  'cc_rec_end_date',  'cc_closed_date_sk',  'cc_open_date_sk',  'cc_name',  'cc_class',  'cc_employees',  'cc_sq_ft',  'cc_hours',  'cc_manager',  'cc_mkt_id',  'cc_mkt_class',  'cc_mkt_desc',  'cc_market_manager',  'cc_division',  'cc_division_name',  'cc_company',  'cc_company_name',  'cc_street_number',  'cc_street_name',  'cc_street_type',  'cc_suite_number',  'cc_city',  'cc_county',  'cc_state',  'cc_zip',  'cc_country'],
             'catalog_page': ['cp_catalog_page_sk',  'cp_catalog_page_id',  'cp_start_date_sk',  'cp_end_date_sk',  'cp_department',  'cp_catalog_number',  'cp_catalog_page_number',  'cp_description',  'cp_type'],
             'catalog_returns': ['cr_returned_date_sk',  'cr_returned_time_sk',  'cr_item_sk',  'cr_refunded_customer_sk',  'cr_refunded_cdemo_sk',  'cr_refunded_hdemo_sk',  'cr_refunded_addr_sk',  'cr_returning_customer_sk',  'cr_returning_cdemo_sk',  'cr_returning_hdemo_sk',  'cr_returning_addr_sk',  'cr_call_center_sk',  'cr_catalog_page_sk',  'cr_ship_mode_sk',  'cr_warehouse_sk',  'cr_reason_sk',  'cr_order_number',  'cr_return_quantity'],
             'catalog_sales': ['cs_sold_date_sk',  'cs_sold_time_sk',  'cs_ship_date_sk',  'cs_bill_customer_sk',  'cs_bill_cdemo_sk',  'cs_bill_hdemo_sk',  'cs_bill_addr_sk',  'cs_ship_customer_sk',  'cs_ship_cdemo_sk',  'cs_ship_hdemo_sk',  'cs_ship_addr_sk',  'cs_call_center_sk',  'cs_catalog_page_sk',  'cs_ship_mode_sk',  'cs_warehouse_sk',  'cs_item_sk',  'cs_promo_sk',  'cs_order_number',  'cs_quantity'],
             'customer': ['c_customer_sk',  'c_customer_id',  'c_current_cdemo_sk',  'c_current_hdemo_sk',  'c_current_addr_sk',  'c_first_shipto_date_sk',  'c_first_sales_date_sk',  'c_salutation',  'c_first_name',  'c_last_name',  'c_preferred_cust_flag',  'c_birth_day',  'c_birth_month',  'c_birth_year',  'c_birth_country',  'c_login',  'c_email_address',  'c_last_review_date_sk'],
             'customer_address': ['ca_address_sk',  'ca_address_id',  'ca_street_number',  'ca_street_name',  'ca_street_type',  'ca_suite_number',  'ca_city',  'ca_county',  'ca_state',  'ca_zip',  'ca_country',  'ca_location_type'],
             'customer_demographics': ['cd_demo_sk',  'cd_gender',  'cd_marital_status',  'cd_education_status',  'cd_purchase_estimate',  'cd_credit_rating',  'cd_dep_count',  'cd_dep_employed_count',  'cd_dep_college_count'],
             'date_dim': ['d_date_sk',  'd_date_id',  'd_date',  'd_month_seq',  'd_week_seq',  'd_quarter_seq',  'd_year',  'd_dow',  'd_moy',  'd_dom',  'd_qoy',  'd_fy_year',  'd_fy_quarter_seq',  'd_fy_week_seq',  'd_day_name',  'd_quarter_name',  'd_holiday',  'd_weekend',  'd_following_holiday',  'd_first_dom',  'd_last_dom',  'd_same_day_ly',  'd_same_day_lq',  'd_current_day',  'd_current_week',  'd_current_month',  'd_current_quarter',  'd_current_year'],
             'dbgen_version': ['dv_version',  'dv_create_date',  'dv_create_time',  'dv_cmdline_args'],
             'household_demographics': ['hd_demo_sk',  'hd_income_band_sk',  'hd_buy_potential',  'hd_dep_count',  'hd_vehicle_count'],
             'income_band': ['ib_income_band_sk', 'ib_lower_bound', 'ib_upper_bound'],
             'inventory': ['inv_date_sk',  'inv_item_sk',  'inv_warehouse_sk',  'inv_quantity_on_hand'],
             'item': ['i_item_sk',  'i_item_id',  'i_rec_start_date',  'i_rec_end_date',  'i_item_desc',  'i_brand_id',  'i_brand',  'i_class_id',  'i_class',  'i_category_id',  'i_category',  'i_manufact_id',  'i_manufact',  'i_size',  'i_formulation',  'i_color',  'i_units',  'i_container',  'i_manager_id',  'i_product_name'],
             'promotion': ['p_promo_sk',  'p_promo_id',  'p_start_date_sk',  'p_end_date_sk',  'p_item_sk',  'p_response_target',  'p_promo_name',  'p_channel_dmail',  'p_channel_email',  'p_channel_catalog',  'p_channel_tv',  'p_channel_radio',  'p_channel_press',  'p_channel_event',  'p_channel_demo',  'p_channel_details',  'p_purpose',  'p_discount_active'],
             'reason': ['r_reason_sk', 'r_reason_id', 'r_reason_desc'],
             'ship_mode': ['sm_ship_mode_sk',  'sm_ship_mode_id',  'sm_type',  'sm_code',  'sm_carrier',  'sm_contract'],
             'store': ['s_store_sk',  's_store_id',  's_rec_start_date',  's_rec_end_date',  's_closed_date_sk',  's_store_name',  's_number_employees',  's_floor_space',  's_hours',  's_manager',  's_market_id',  's_geography_class',  's_market_desc',  's_market_manager',  's_division_id',  's_division_name',  's_company_id',  's_company_name',  's_street_number',  's_street_name',  's_street_type',  's_suite_number',  's_city',  's_county',  's_state',  's_zip',  's_country'],
             'store_returns': ['sr_returned_date_sk',  'sr_return_time_sk',  'sr_item_sk',  'sr_customer_sk',  'sr_cdemo_sk',  'sr_hdemo_sk',  'sr_addr_sk',  'sr_store_sk',  'sr_reason_sk',  'sr_ticket_number',  'sr_return_quantity'],
             'store_sales': ['ss_sold_date_sk',  'ss_sold_time_sk',  'ss_item_sk',  'ss_customer_sk',  'ss_cdemo_sk',  'ss_hdemo_sk',  'ss_addr_sk',  'ss_store_sk',  'ss_promo_sk',  'ss_ticket_number',  'ss_quantity'],
             'time_dim': ['t_time_sk',  't_time_id',  't_time',  't_hour',  't_minute',  't_second',  't_am_pm',  't_shift',  't_sub_shift',  't_meal_time'],
             'warehouse': ['w_warehouse_sk',  'w_warehouse_id',  'w_warehouse_name',  'w_warehouse_sq_ft',  'w_street_number',  'w_street_name',  'w_street_type',  'w_suite_number',  'w_city',  'w_county',  'w_state',  'w_zip',  'w_country'],
             'web_page': ['wp_web_page_sk',  'wp_web_page_id',  'wp_rec_start_date',  'wp_rec_end_date',  'wp_creation_date_sk',  'wp_access_date_sk',  'wp_autogen_flag',  'wp_customer_sk',  'wp_url',  'wp_type',  'wp_char_count',  'wp_link_count',  'wp_image_count',  'wp_max_ad_count'],
             'web_returns': ['wr_returned_date_sk',  'wr_returned_time_sk',  'wr_item_sk',  'wr_refunded_customer_sk',  'wr_refunded_cdemo_sk',  'wr_refunded_hdemo_sk',  'wr_refunded_addr_sk',  'wr_returning_customer_sk',  'wr_returning_cdemo_sk',  'wr_returning_hdemo_sk',  'wr_returning_addr_sk',  'wr_web_page_sk',  'wr_reason_sk',  'wr_order_number',  'wr_return_quantity'],
             'web_sales': ['ws_sold_date_sk',  'ws_sold_time_sk',  'ws_ship_date_sk',  'ws_item_sk',  'ws_bill_customer_sk',  'ws_bill_cdemo_sk',  'ws_bill_hdemo_sk',  'ws_bill_addr_sk',  'ws_ship_customer_sk',  'ws_ship_cdemo_sk',  'ws_ship_hdemo_sk',  'ws_ship_addr_sk',  'ws_web_page_sk',  'ws_web_site_sk',  'ws_ship_mode_sk',  'ws_warehouse_sk',  'ws_promo_sk',  'ws_order_number',  'ws_quantity'],
             'web_site': ['web_site_sk',  'web_site_id',  'web_rec_start_date',  'web_rec_end_date',  'web_name',  'web_open_date_sk',  'web_close_date_sk',  'web_class',  'web_manager',  'web_mkt_id',  'web_mkt_class',  'web_mkt_desc',  'web_market_manager',  'web_company_id',  'web_company_name',  'web_street_number',  'web_street_name',  'web_street_type',  'web_suite_number',  'web_city',  'web_county',  'web_state',  'web_zip',  'web_country']

        })
    RANGE_COLUMNS = collections.defaultdict(
        list,
        {
            'call_center': ['cc_gmt_offset', 'cc_tax_percentage'],
             'catalog_returns': ['cr_return_amount',  'cr_return_tax',  'cr_return_amt_inc_tax',  'cr_fee',  'cr_return_ship_cost',  'cr_refunded_cash',  'cr_reversed_charge',  'cr_store_credit',  'cr_net_loss'],
             'catalog_sales': ['cs_wholesale_cost',  'cs_list_price',  'cs_sales_price',  'cs_ext_discount_amt',  'cs_ext_sales_price',  'cs_ext_wholesale_cost',  'cs_ext_list_price',  'cs_ext_tax',  'cs_coupon_amt',  'cs_ext_ship_cost',  'cs_net_paid',  'cs_net_paid_inc_tax',  'cs_net_paid_inc_ship',  'cs_net_paid_inc_ship_tax',  'cs_net_profit'],
             'customer_address': ['ca_gmt_offset'],
             'item': ['i_current_price', 'i_wholesale_cost'],
             'promotion': ['p_cost'],
             'store': ['s_gmt_offset', 's_tax_precentage'],
             'store_returns': ['sr_return_amt',  'sr_return_tax',  'sr_return_amt_inc_tax',  'sr_fee',  'sr_return_ship_cost',  'sr_refunded_cash',  'sr_reversed_charge',  'sr_store_credit',  'sr_net_loss'],
             'store_sales': ['ss_wholesale_cost',  'ss_list_price',  'ss_sales_price',  'ss_ext_discount_amt',  'ss_ext_sales_price',  'ss_ext_wholesale_cost',  'ss_ext_list_price',  'ss_ext_tax',  'ss_coupon_amt',  'ss_net_paid',  'ss_net_paid_inc_tax',  'ss_net_profit'],
             'warehouse': ['w_gmt_offset'],
             'web_returns': ['wr_return_amt',  'wr_return_tax',  'wr_return_amt_inc_tax',  'wr_fee',  'wr_return_ship_cost',  'wr_refunded_cash',  'wr_reversed_charge',  'wr_account_credit',  'wr_net_loss'],
             'web_sales': ['ws_wholesale_cost',  'ws_list_price',  'ws_sales_price',  'ws_ext_discount_amt',  'ws_ext_sales_price',  'ws_ext_wholesale_cost',  'ws_ext_list_price',  'ws_ext_tax',  'ws_coupon_amt',  'ws_ext_ship_cost',  'ws_net_paid',  'ws_net_paid_inc_tax',  'ws_net_paid_inc_ship',  'ws_net_paid_inc_ship_tax',  'ws_net_profit'],
             'web_site': ['web_gmt_offset', 'web_tax_percentage']

        })

    CSV_FILES = ['store.csv', 'inventory.csv','ship_mode.csv',
         'income_band.csv','catalog_returns.csv','reason.csv',
         'time_dim.csv','customer_address.csv','catalog_sales.csv',
         'web_sales.csv','date_dim.csv','promotion.csv','web_returns.csv',
         'call_center.csv','item.csv','catalog_page.csv','web_site.csv',
         'customer_demographics.csv', 'household_demographics.csv',
         'customer.csv','web_page.csv','store_sales.csv','store_returns.csv',
         'dbgen_version.csv','warehouse.csv'
    ]

    BASE_TABLE_PRED_COLS = collections.defaultdict(
        list,
        {
            'catalog_returns': ['cr_returned_date_sk',  'cr_returned_time_sk',  'cr_item_sk',  'cr_refunded_customer_sk',  'cr_refunded_cdemo_sk',  'cr_refunded_hdemo_sk',  'cr_refunded_addr_sk',  'cr_returning_customer_sk',  'cr_returning_cdemo_sk',  'cr_returning_hdemo_sk',  'cr_returning_addr_sk',  'cr_call_center_sk',  'cr_catalog_page_sk',  'cr_ship_mode_sk',  'cr_warehouse_sk',  'cr_reason_sk',  'cr_order_number',  'cr_return_quantity'],
            'catalog_sales': ['cs_sold_date_sk',  'cs_sold_time_sk',  'cs_ship_date_sk',  'cs_bill_customer_sk',  'cs_bill_cdemo_sk',  'cs_bill_hdemo_sk',  'cs_bill_addr_sk',  'cs_ship_customer_sk',  'cs_ship_cdemo_sk',  'cs_ship_hdemo_sk',  'cs_ship_addr_sk',  'cs_call_center_sk',  'cs_catalog_page_sk',  'cs_ship_mode_sk',  'cs_warehouse_sk',  'cs_item_sk',  'cs_promo_sk',  'cs_order_number',  'cs_quantity'],
            'inventory': ['inv_date_sk',  'inv_item_sk',  'inv_warehouse_sk',  'inv_quantity_on_hand'],
            'item': ['i_item_sk',  'i_item_id',  'i_rec_start_date',  'i_rec_end_date',  'i_item_desc',  'i_brand_id',  'i_brand',  'i_class_id',  'i_class',  'i_category_id',  'i_category',  'i_manufact_id',  'i_manufact',  'i_size',  'i_formulation',  'i_color',  'i_units',  'i_container',  'i_manager_id',  'i_product_name'],
            'promotion': ['p_promo_sk',  'p_promo_id',  'p_start_date_sk',  'p_end_date_sk',  'p_item_sk',  'p_response_target',  'p_promo_name',  'p_channel_dmail',  'p_channel_email',  'p_channel_catalog',  'p_channel_tv',  'p_channel_radio',  'p_channel_press',  'p_channel_event',  'p_channel_demo',  'p_channel_details',  'p_purpose',  'p_discount_active'],
            'store_returns': ['sr_returned_date_sk',  'sr_return_time_sk',  'sr_item_sk',  'sr_customer_sk',  'sr_cdemo_sk',  'sr_hdemo_sk',  'sr_addr_sk',  'sr_store_sk',  'sr_reason_sk',  'sr_ticket_number',  'sr_return_quantity'],
            'store_sales': ['ss_sold_date_sk',  'ss_sold_time_sk',  'ss_item_sk',  'ss_customer_sk',  'ss_cdemo_sk',  'ss_hdemo_sk',  'ss_addr_sk',  'ss_store_sk',  'ss_promo_sk',  'ss_ticket_number',  'ss_quantity'],
            'web_returns': ['wr_returned_date_sk',  'wr_returned_time_sk',  'wr_item_sk',  'wr_refunded_customer_sk',  'wr_refunded_cdemo_sk',  'wr_refunded_hdemo_sk',  'wr_refunded_addr_sk',  'wr_returning_customer_sk',  'wr_returning_cdemo_sk',  'wr_returning_hdemo_sk',  'wr_returning_addr_sk',  'wr_web_page_sk',  'wr_reason_sk',  'wr_order_number',  'wr_return_quantity'],
            'web_sales': ['ws_sold_date_sk',  'ws_sold_time_sk',  'ws_ship_date_sk',  'ws_item_sk',  'ws_bill_customer_sk',  'ws_bill_cdemo_sk',  'ws_bill_hdemo_sk',  'ws_bill_addr_sk',  'ws_ship_customer_sk',  'ws_ship_cdemo_sk',  'ws_ship_hdemo_sk',  'ws_ship_addr_sk',  'ws_web_page_sk',  'ws_web_site_sk',  'ws_ship_mode_sk',  'ws_warehouse_sk',  'ws_promo_sk',  'ws_order_number',  'ws_quantity'],

        })

    TDS_M_PRED_COLS = collections.defaultdict(
        list, {
            'catalog_returns': ['cr_returned_date_sk',  'cr_returned_time_sk',  'cr_item_sk',  'cr_refunded_customer_sk',  'cr_refunded_cdemo_sk',  'cr_refunded_hdemo_sk',  'cr_refunded_addr_sk',  'cr_returning_customer_sk',  'cr_returning_cdemo_sk',  'cr_returning_hdemo_sk',  'cr_returning_addr_sk',  'cr_call_center_sk',  'cr_catalog_page_sk',  'cr_ship_mode_sk',  'cr_warehouse_sk',  'cr_reason_sk',  'cr_order_number',  'cr_return_quantity'],
            'catalog_sales': ['cs_sold_date_sk',  'cs_sold_time_sk',  'cs_ship_date_sk',  'cs_bill_customer_sk',  'cs_bill_cdemo_sk',  'cs_bill_hdemo_sk',  'cs_bill_addr_sk',  'cs_ship_customer_sk',  'cs_ship_cdemo_sk',  'cs_ship_hdemo_sk',  'cs_ship_addr_sk',  'cs_call_center_sk',  'cs_catalog_page_sk',  'cs_ship_mode_sk',  'cs_warehouse_sk',  'cs_item_sk',  'cs_promo_sk',  'cs_order_number',  'cs_quantity'],
            'inventory': ['inv_date_sk',  'inv_item_sk',  'inv_warehouse_sk',  'inv_quantity_on_hand'],
            'item': ['i_item_sk',  'i_item_id',  'i_rec_start_date',  'i_rec_end_date',  'i_item_desc',  'i_brand_id',  'i_brand',  'i_class_id',  'i_class',  'i_category_id',  'i_category',  'i_manufact_id',  'i_manufact',  'i_size',  'i_formulation',  'i_color',  'i_units',  'i_container',  'i_manager_id',  'i_product_name'],
            'promotion': ['p_promo_sk',  'p_promo_id',  'p_start_date_sk',  'p_end_date_sk',  'p_item_sk',  'p_response_target',  'p_promo_name',  'p_channel_dmail',  'p_channel_email',  'p_channel_catalog',  'p_channel_tv',  'p_channel_radio',  'p_channel_press',  'p_channel_event',  'p_channel_demo',  'p_channel_details',  'p_purpose',  'p_discount_active'],
            'store_returns': ['sr_returned_date_sk',  'sr_return_time_sk',  'sr_item_sk',  'sr_customer_sk',  'sr_cdemo_sk',  'sr_hdemo_sk',  'sr_addr_sk',  'sr_store_sk',  'sr_reason_sk',  'sr_ticket_number',  'sr_return_quantity'],
            'store_sales': ['ss_sold_date_sk',  'ss_sold_time_sk',  'ss_item_sk',  'ss_customer_sk',  'ss_cdemo_sk',  'ss_hdemo_sk',  'ss_addr_sk',  'ss_store_sk',  'ss_promo_sk',  'ss_ticket_number',  'ss_quantity'],
            'web_returns': ['wr_returned_date_sk',  'wr_returned_time_sk',  'wr_item_sk',  'wr_refunded_customer_sk',  'wr_refunded_cdemo_sk',  'wr_refunded_hdemo_sk',  'wr_refunded_addr_sk',  'wr_returning_customer_sk',  'wr_returning_cdemo_sk',  'wr_returning_hdemo_sk',  'wr_returning_addr_sk',  'wr_web_page_sk',  'wr_reason_sk',  'wr_order_number',  'wr_return_quantity'],
            'web_sales': ['ws_sold_date_sk',  'ws_sold_time_sk',  'ws_ship_date_sk',  'ws_item_sk',  'ws_bill_customer_sk',  'ws_bill_cdemo_sk',  'ws_bill_hdemo_sk',  'ws_bill_addr_sk',  'ws_ship_customer_sk',  'ws_ship_cdemo_sk',  'ws_ship_hdemo_sk',  'ws_ship_addr_sk',  'ws_web_page_sk',  'ws_web_site_sk',  'ws_ship_mode_sk',  'ws_warehouse_sk',  'ws_promo_sk',  'ws_order_number',  'ws_quantity'],

        })

    TDS_FULL_PRED_COLS = collections.defaultdict(
        list, {
            'catalog_returns': ['cr_returned_date_sk',  'cr_returned_time_sk',  'cr_item_sk',  'cr_refunded_customer_sk',  'cr_refunded_cdemo_sk',  'cr_refunded_hdemo_sk',  'cr_refunded_addr_sk',  'cr_returning_customer_sk',  'cr_returning_cdemo_sk',  'cr_returning_hdemo_sk',  'cr_returning_addr_sk',  'cr_call_center_sk',  'cr_catalog_page_sk',  'cr_ship_mode_sk',  'cr_warehouse_sk',  'cr_reason_sk',  'cr_order_number',  'cr_return_quantity'],
            'catalog_sales': ['cs_sold_date_sk',  'cs_sold_time_sk',  'cs_ship_date_sk',  'cs_bill_customer_sk',  'cs_bill_cdemo_sk',  'cs_bill_hdemo_sk',  'cs_bill_addr_sk',  'cs_ship_customer_sk',  'cs_ship_cdemo_sk',  'cs_ship_hdemo_sk',  'cs_ship_addr_sk',  'cs_call_center_sk',  'cs_catalog_page_sk',  'cs_ship_mode_sk',  'cs_warehouse_sk',  'cs_item_sk',  'cs_promo_sk',  'cs_order_number',  'cs_quantity'],
            'inventory': ['inv_date_sk',  'inv_item_sk',  'inv_warehouse_sk',  'inv_quantity_on_hand'],
            'item': ['i_item_sk',  'i_item_id',  'i_rec_start_date',  'i_rec_end_date',  'i_item_desc',  'i_brand_id',  'i_brand',  'i_class_id',  'i_class',  'i_category_id',  'i_category',  'i_manufact_id',  'i_manufact',  'i_size',  'i_formulation',  'i_color',  'i_units',  'i_container',  'i_manager_id',  'i_product_name'],
            'promotion': ['p_promo_sk',  'p_promo_id',  'p_start_date_sk',  'p_end_date_sk',  'p_item_sk',  'p_response_target',  'p_promo_name',  'p_channel_dmail',  'p_channel_email',  'p_channel_catalog',  'p_channel_tv',  'p_channel_radio',  'p_channel_press',  'p_channel_event',  'p_channel_demo',  'p_channel_details',  'p_purpose',  'p_discount_active'],
            'store_returns': ['sr_returned_date_sk',  'sr_return_time_sk',  'sr_item_sk',  'sr_customer_sk',  'sr_cdemo_sk',  'sr_hdemo_sk',  'sr_addr_sk',  'sr_store_sk',  'sr_reason_sk',  'sr_ticket_number',  'sr_return_quantity'],
            'store_sales': ['ss_sold_date_sk',  'ss_sold_time_sk',  'ss_item_sk',  'ss_customer_sk',  'ss_cdemo_sk',  'ss_hdemo_sk',  'ss_addr_sk',  'ss_store_sk',  'ss_promo_sk',  'ss_ticket_number',  'ss_quantity'],
            'web_returns': ['wr_returned_date_sk',  'wr_returned_time_sk',  'wr_item_sk',  'wr_refunded_customer_sk',  'wr_refunded_cdemo_sk',  'wr_refunded_hdemo_sk',  'wr_refunded_addr_sk',  'wr_returning_customer_sk',  'wr_returning_cdemo_sk',  'wr_returning_hdemo_sk',  'wr_returning_addr_sk',  'wr_web_page_sk',  'wr_reason_sk',  'wr_order_number',  'wr_return_quantity'],
            'web_sales': ['ws_sold_date_sk',  'ws_sold_time_sk',  'ws_ship_date_sk',  'ws_item_sk',  'ws_bill_customer_sk',  'ws_bill_cdemo_sk',  'ws_bill_hdemo_sk',  'ws_bill_addr_sk',  'ws_ship_customer_sk',  'ws_ship_cdemo_sk',  'ws_ship_hdemo_sk',  'ws_ship_addr_sk',  'ws_web_page_sk',  'ws_web_site_sk',  'ws_ship_mode_sk',  'ws_warehouse_sk',  'ws_promo_sk',  'ws_order_number',  'ws_quantity'],

        })

    # For JOB-light schema.
    TRUE_FULL_OUTER_CARDINALITY = {

    }
    _CONTENT_COLS = None

    @staticmethod
    def ContentColumns():
        if TPC_DS._CONTENT_COLS is None:
            TPC_DS._CONTENT_COLS = {
                '{}.csv'.format(table_name):
                range_cols + TPC_DS.CATEGORICAL_COLUMNS[table_name]
                for table_name, range_cols in
                TPC_DS.RANGE_COLUMNS.items()
            }
            # Add join keys.
            for table_name in TPC_DS._CONTENT_COLS:
                cols = TPC_DS._CONTENT_COLS[table_name]
                if table_name == 'item.csv':
                    cols.append('i_item_sk')
                elif  table_name == 'sales_store.csv' : 
                    cols.append('ss_item_sk')
                elif  table_name == 'store_returns.csv' : 
                    cols.append('sr_item_sk')
                elif  table_name == 'catalog_sales.csv' : 
                    cols.append('cs_item_sk') 
                elif  table_name == 'catalog_returns.csv' : 
                    cols.append('cr_item_sk') 
                elif  table_name == 'web_sales.csv' : 
                    cols.append('ws_item_sk') 
                elif  table_name == 'web_returns.csv' : 
                    cols.append('wr_item_sk') 
                elif  table_name == 'inventory.csv' : 
                    cols.append('inv_item_sk') 
                elif  table_name == 'promotion.csv' : 
                    cols.append('p_item_sk')

        return TPC_DS._CONTENT_COLS

    @staticmethod
    def GetFullOuterCardinalityOrFail(join_tables):
        key = tuple(sorted(join_tables))
        return TPC_DS.TRUE_FULL_OUTER_CARDINALITY[key]

    @staticmethod
    def GetTDSLightJoinKeys():
        return {
            'item.csv' : 'i_item_sk',
            'sales_store.csv' : 'ss_item_sk',
            'store_returns.csv' : 'sr_item_sk',
            'catalog_sales.csv' : 'cs_item_sk', 
            'catalog_returns.csv' : 'cr_item_sk', 
            'web_sales.csv' : 'ws_item_sk', 
            'web_returns.csv' : 'wr_item_sk', 
            'inventory.csv' : 'inv_item_sk', 
            'promotion.csv' : 'p_item_sk'
        }


def LoadImdb(table=None,
             data_dir='./datasets/job/',
             try_load_parsed=True,
             use_cols='simple'):
    """Loads IMDB tables with a specified set of columns.

    use_cols:
      simple: only movie_id join keys (JOB-light)
      content: + content columns (JOB-light-ranges)
      multi: all join keys in JOB-M
      full: all join keys in JOB-full
      None: load all columns

    Returns:
      A single CsvTable if 'table' is specified, else a dict of CsvTables.
    """
    assert use_cols in ['simple', 'content', 'multi', 'full', None], use_cols

    def TryLoad(table_name, filepath, use_cols, **kwargs):
        """Try load from previously parsed (table, columns)."""
        if use_cols:
            cols_str = '-'.join(use_cols)
            parsed_path = filepath[:-4] + '.{}.table'.format(cols_str)
        else:
            parsed_path = filepath[:-4] + '.table'
        if try_load_parsed:
            if os.path.exists(parsed_path):
                arr = np.load(parsed_path, allow_pickle=True)
                print('Loaded parsed Table from', parsed_path)
                table = arr.item()
                print(table)
                return table
        table = CsvTable(
            table_name,
            filepath,
            cols=use_cols,
            **kwargs,
        )
        if try_load_parsed:
            np.save(open(parsed_path, 'wb'), table)
            print('Saved parsed Table to', parsed_path)
        return table

    def get_use_cols(filepath):
        if use_cols == 'simple':
            return JoinOrderBenchmark.BASE_TABLE_PRED_COLS.get(filepath, None)
        elif use_cols == 'content':
            return JoinOrderBenchmark.ContentColumns().get(filepath, None)
        elif use_cols == 'multi':
            return JoinOrderBenchmark.JOB_M_PRED_COLS.get(filepath, None)
        elif use_cols == 'full':
            return JoinOrderBenchmark.JOB_FULL_PRED_COLS.get(filepath, None)
        return None  # Load all.

    if table:
        filepath = table + '.csv'
        table = TryLoad(
            table,
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
            escapechar='\\',
        )
        return table

    tables = {}
    for filepath in JoinOrderBenchmark.BASE_TABLE_PRED_COLS:
        tables[filepath[0:-4]] = TryLoad(
            filepath[0:-4],
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
            escapechar='\\',
        )

    return tables


def LoadTds(table=None,
             data_dir='./datasets/tds/',
             try_load_parsed=True,
             use_cols='simple'):
    """Loads IMDB tables with a specified set of columns.

    use_cols:
      simple: only movie_id join keys (JOB-light)
      content: + content columns (JOB-light-ranges)
      multi: all join keys in JOB-M
      full: all join keys in JOB-full
      None: load all columns

    Returns:
      A single CsvTable if 'table' is specified, else a dict of CsvTables.
    """
    assert use_cols in ['simple', 'content', 'multi', 'full', None], use_cols

    def TryLoad(table_name, filepath, use_cols, **kwargs):
        """Try load from previously parsed (table, columns)."""
        if use_cols:
            cols_str = '-'.join(use_cols)
            parsed_path = filepath[:-4] + '.{}.table'.format(cols_str)
        else:
            parsed_path = filepath[:-4] + '.table'
        if try_load_parsed:
            if os.path.exists(parsed_path):
                arr = np.load(parsed_path, allow_pickle=True)
                print('Loaded parsed Table from', parsed_path)
                table = arr.item()
                print(table)
                return table
        table = CsvTable(
            table_name,
            filepath,
            cols=use_cols,
            **kwargs,
        )
        if try_load_parsed:
            np.save(open(parsed_path, 'wb'), table)
            print('Saved parsed Table to', parsed_path)
        return table

    def get_use_cols(filepath):
        if use_cols == 'simple':
            return TPC_DS.BASE_TABLE_PRED_COLS.get(filepath, None)
        elif use_cols == 'content':
            return TPC_DS.ContentColumns().get(filepath, None)
        elif use_cols == 'multi':
            return TPC_DS.TDS_M_PRED_COLS.get(filepath, None)
        elif use_cols == 'full':
            return TPC_DS.TDS_FULL_PRED_COLS.get(filepath, None)
        return None  # Load all.

    if table:
        filepath = table + '.csv'
        table = TryLoad(
            table,
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
            escapechar='\\',
        )
        return table

    tables = {}
    for filepath in TPC_DS.BASE_TABLE_PRED_COLS:
        tables[filepath[0:-4]] = TryLoad(
            filepath[0:-4],
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
            escapechar='\\',
        )

    return tables