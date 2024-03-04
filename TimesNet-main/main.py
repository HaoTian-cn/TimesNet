from EXP import *
if __name__ == '__main__':

    for ii in range(args.itr):
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_eb{}_{}'.format(
            args.task_name,
            args.model,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.embed,
            ii)
        exp = Exp_Long_Term_Forecast(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
