import Settings
import utils.htmlprocessor as uhp
import utils.session as us
import utils.research as ur

from model import setup as ms

file = Settings.WEB_DATA_DIR / 'current_research_2.html'

if __name__ == "__main__":
    pbs_data_df = uhp.PBS_DATA
    print()

    pp = uhp.HtmlProcessor(html_parser='html5', html_filepath=file, seed=706271.76)
    # pp.process_html_predictions()  
    # pp.pickle_save(override=True)
    
    print(uhp.create_market_volume_df())

    # pp.pickle_save()

    # pp.view_all_results(info=True)

    # print(len(d.find_all('tr')))


    plort = 'Dervish'
    # print(pp.results['recsales'])

    xcols = ['Plort Value']


    maf = ms.MarketAnalysisFormatter()
    # session = maf.start_session(
    #     session_type='PP', x_columns=xcols,
    #     plort=plort, target_column='Sell'
    # )
    # set_type = 'Training'
    # session.run_model(
    #     set_type=set_type,
    # )    
    # session.get_results(set_type)



    # period = 10
    # combined_df = maf.combine_plort_dfs(plort)
    # maf.extract_local_extremes(plort, period, chunk_review_limit=10)
    # ms.create_basic_plort_plot(combined_df, plort)
    
    # tls, cvls = session.determine_best_poly_features(10)
    # session.dbpf_visualize(tls, cvls)
    
    # model,r = session.run_prediction('Training')
    
    # print(model)
    # session.run_prediction('CV')
    # preds = session.get_results('Training')
    # preds = session.get_results('CV')


    

    # s = us.Session()
    # print(s.market_volume)

    

    # index = 0

    # all_values = pp.results['plain'][plort]
    # all_stdevs = pp.results['stdev'][plort]

    # value = int(all_values[index])
    
    # stdev = all_values.std()
    # mean = all_values.mean()

    # z_score = (value - mean) / stdev
    # max_value = all_values.max()

    # print(f'\t{plort} - day {index+1}')
    # print(f'value: {value}')
    # print(f'z score: {z_score}')
    # print(f'reported as {all_stdevs[index]}% of best')
    # print(f'value percentage of maximum in 999 days: {value / max_value }')




    # print('Days until empty:')
    # ddd = data_df.loc[:, ['Saturation Value', 'Plort Name']].to_dict(orient='index')
    # for index in ddd:
    #     days_til_empty = ur.days_until_amount(0, ddd[index]['Saturation Value'] / 2)
    #     print(f'\t{ddd[index]['Plort Name']:10s} - {days_til_empty}')


    


