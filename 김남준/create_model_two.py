import my_package as my

path = 'C:/Users/user/PycharmProjects/AVA/dataset/'
train_file_names, test_file_names = my.util.load_file_names(path)

my.util.print_process('데이터전처리(two) 실행')
train_df2 = my.preprocessing_two.jsonl2df(train_file_names)
test_df2 = my.preprocessing_two.jsonl2df(test_file_names)
train_df2, test_df2, props2 = my.preprocessing_two.set_max_length_for_train(train_df2, test_df2, max_length=300)
my.util.save(path, 2, train_df2, test_df2, props2)

my.util.print_process('프로그램 종료')
