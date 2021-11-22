import my_package as my

path = 'C:/Users/user/PycharmProjects/AVA/dataset/'
train_file_names, test_file_names = my.util.load_file_names(path)

my.util.print_process('데이터전처리(one) 실행')
train_df1 = my.preprocessing_one.jsonl2df(file_names=train_file_names)
test_df1 = my.preprocessing_one.jsonl2df(file_names=test_file_names)
train_df1, props1 = my.preprocessing_one.reduce_features_for_train(df=train_df1, n_pca=9)
test_df1 = my.preprocessing_one.reduce_features_for_test(df=test_df1, props=props1)
my.util.save(path, 1, train_df1, test_df1, props1)

my.util.print_process('모델링(one) 실행')
model1 = my.modeling_one.create_random_forest_model(
    train_df=train_df1,
    test_df=test_df1
)
my.util.save(path, 3, model1)

my.util.print_process('프로그램 종료')
