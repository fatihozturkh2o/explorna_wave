#!pip install spektral -q
from spektral.layers import GraphConv, GraphConvSkip, GatedGraphConv, GraphAttention, ARMAConv,ChebConv
import pandas as pd
import numpy as np
import tensorflow as tf
import gc
import os

# All related functions and default parameters.

token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}
sequence_token2int = {x: i for i, x in enumerate('AGUC')}
structure_token2int = {
    '.': 0,
    '(': 1,
    ')': 2,
}
loop_token2int = {x: i for i, x in enumerate('SMIBHEX')}
token2int_map = {
    "sequence": sequence_token2int,
    "structure": structure_token2int,
    "predicted_loop_type": loop_token2int
}

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
pred_cols_2 = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
               'deg_error_Mg_50C', 'deg_error_50C']


def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    base_fea = np.transpose(
        np.array(
            df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
        ),
        (0, 2, 1)
    )

    base_fea_ = np.transpose(
        np.array([
            df[col]
                .apply(lambda seq: [token2int_map[col][x] for x in seq])
                .values
                .tolist()
            for col in cols
        ]),
        (1, 2, 0)
    )

    bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:, :, np.newaxis]
    bpps_max_fea = np.array(df['bpps_max'].to_list())[:, :, np.newaxis]

    bpps_sum_fea = bpps_sum_fea[:, :base_fea.shape[1], :]
    bpps_max_fea = bpps_max_fea[:, :base_fea.shape[1], :]

    ohe_1 = tf.keras.utils.to_categorical(base_fea_[:, :, 0], 4)
    ohe_2 = tf.keras.utils.to_categorical(base_fea_[:, :, 1], 3)
    ohe_3 = tf.keras.utils.to_categorical(base_fea_[:, :, 2], 7)

    return np.concatenate([base_fea, bpps_sum_fea, bpps_max_fea], 2), np.concatenate([ohe_1, ohe_2, ohe_3], axis=2)


def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return tf.math.sqrt(mse)


def mcrmse(y_actual, y_pred, num_scored=len(pred_cols)):
    score = 0
    for i in range(num_scored):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
    return score


def mcrmse_weighted_by_pos(y_actual, y_pred):
    score = 0
    pre_length = 5
    for i in range(5):
        prefix_s = tf.keras.losses.mean_squared_error(y_actual[:, :pre_length, i],
                                                      y_pred[:, :pre_length, i]) * pre_length
        body_s = tf.keras.losses.mean_squared_error(y_actual[:, pre_length:, i], y_pred[:, pre_length:, i]) * (
                    68 - pre_length)
        score += tf.math.sqrt((prefix_s + body_s * 4) / 68) / 5.0
    return score


def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(tf.keras.layers.GRU(hidden_dim, dropout=dropout, return_sequences=True,
                                                             kernel_initializer='orthogonal'),
                                         merge_mode="concat", )


def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, dropout=dropout, return_sequences=True,
                                                              kernel_initializer='orthogonal'),
                                         merge_mode="concat", )


def build_base_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=256, type=0, ):
    def wave_layer(x):
        def wave_block(x, filters, kernel_size, n):
            dilation_rates = [2 ** i for i in range(n)]
            x = tf.keras.layers.Conv1D(filters=filters,
                                       kernel_size=1,
                                       padding='same')(x)
            res_x = x
            for dilation_rate in dilation_rates:
                tanh_out = tf.keras.layers.Conv1D(filters=filters,
                                                  kernel_size=kernel_size,
                                                  padding='same',
                                                  activation='tanh',
                                                  dilation_rate=dilation_rate)(x)
                sigm_out = tf.keras.layers.Conv1D(filters=filters,
                                                  kernel_size=kernel_size,
                                                  padding='same',
                                                  activation='sigmoid',
                                                  dilation_rate=dilation_rate)(x)
                x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
                x = tf.keras.layers.Conv1D(filters=filters,
                                           kernel_size=1,
                                           padding='same')(x)
                res_x = tf.keras.layers.Add()([res_x, x])
            return res_x

        x = wave_block(x, 32, 3, 8)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = wave_block(x, 64, 3, 4)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = wave_block(x, 128, 3, 1)
        x = tf.keras.layers.Dropout(0.1)(x)

        return x

    def conv_layer(row, col, x):
        conv = tf.keras.layers.Conv1D(hidden_dim * 2, 5,
                                      padding='same',
                                      activation='tanh',
                                      input_shape=(row, col))(x)

        gcn_1 = GraphConv(
            graph_channels,
            activation='tanh',
        )([conv, As_in[:, :, :, 1]])

        gcn_2 = ChebConv(
            graph_channels,
            activation='tanh',
        )([conv, As_in[:, :, :, 1]])

        gcn_3 = ARMAConv(
            graph_channels,
            activation='tanh',
        )([conv, As_in[:, :, :, 1]])

        gcn_1 = tf.keras.layers.Concatenate()([gcn_1, gcn_2, gcn_3])
        gcn_1 = tf.keras.layers.Conv1D(3 * graph_channels, 5,
                                       padding='same',
                                       activation='tanh',
                                       input_shape=(row, col))(gcn_1)

        conv = tf.keras.layers.Concatenate()([x, conv, gcn_1, gcn_2, gcn_3])
        conv = tf.keras.layers.Activation("relu")(conv)
        conv = tf.keras.layers.SpatialDropout1D(0.1)(conv)

        return conv

    graph_channels = 80 * 1

    inputs = tf.keras.layers.Input(shape=(seq_len, 5))
    ohe_inputs = tf.keras.layers.Input(shape=(seq_len, 14))
    As_in = tf.keras.layers.Input((seq_len, seq_len, 2))

    # split categorical and numerical features and concatenate them later.
    categorical_feat_dim = 3
    categorical_fea = ohe_inputs
    numerical_fea = inputs[:, :, 3:]

    reshaped = tf.keras.layers.concatenate([categorical_fea, numerical_fea], axis=2)

    if type == 0:
        reshaped = tf.keras.layers.concatenate([
            conv_layer(reshaped.shape[1], reshaped.shape[2], reshaped),
            wave_layer(reshaped)
        ], axis=2)
        hidden_1 = gru_layer(hidden_dim, dropout)(reshaped)

        hidden_2 = tf.keras.layers.concatenate([
            conv_layer(hidden_1.shape[1], hidden_1.shape[2], hidden_1),
            wave_layer(hidden_1)
        ], axis=2)

        hidden_3 = gru_layer(hidden_dim, dropout)(hidden_2)
    elif type == 1:
        reshaped = tf.keras.layers.concatenate([
            conv_layer(reshaped.shape[1], reshaped.shape[2], reshaped),
            wave_layer(reshaped)
        ], axis=2)
        hidden_1 = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden_2 = tf.keras.layers.concatenate([
            conv_layer(hidden_1.shape[1], hidden_1.shape[2], hidden_1),
            wave_layer(hidden_1)
        ], axis=2)
        hidden_3 = gru_layer(hidden_dim, dropout)(hidden_2)

    base_model = tf.keras.Model(inputs=[inputs, ohe_inputs, As_in], outputs=hidden_3)
    return base_model


def build_model(base_model, seq_len=107, pred_len=68, type=0, for_pretrain=False):
    inputs = tf.keras.layers.Input(shape=(seq_len, 5))
    ohe_inputs = tf.keras.layers.Input(shape=(seq_len, 14))
    As_in = tf.keras.layers.Input((seq_len, seq_len, 2))

    if for_pretrain:
        truncated = base_model([
            tf.keras.layers.SpatialDropout1D(0.3)(inputs),
            tf.keras.layers.SpatialDropout1D(0.3)(ohe_inputs),
            As_in], )

        out = tf.keras.layers.Dense(14, activation='sigmoid', name="targets")(truncated)

        out = - tf.reduce_mean(
            14 * ohe_inputs * tf.math.log(out + 1e-4) + (1 - ohe_inputs) * tf.math.log(1 - out + 1e-4))

        model = tf.keras.Model(inputs=[inputs, ohe_inputs, As_in], outputs=out)
        model.compile(tf.keras.optimizers.Adam(),
                      loss=lambda t, y: y)
    else:
        truncated = base_model([inputs, ohe_inputs, As_in])[:, :pred_len]
        truncated = tf.keras.layers.concatenate([truncated, inputs[:, :pred_len, -5:]], axis=2)
        out = tf.keras.layers.Dense(5, activation='linear', name="targets")(truncated)
        clf_out = tf.keras.layers.Dense(5, activation='sigmoid', name="outlier")(truncated)

        model = tf.keras.Model(inputs=[inputs, ohe_inputs, As_in], outputs=[out])
        model.compile(tf.keras.optimizers.Adam(),
                      loss=[mcrmse_weighted_by_pos],
                      metrics=[mcrmse],
                      loss_weights=[1, 0])
    return model


def get_adjacency_matrix(inps):
    As = []
    for row in range(0, inps.shape[0]):
        A = np.zeros((inps.shape[1], inps.shape[1]))
        stack = []
        opened_so_far = []

        for seqpos in range(0, inps.shape[1]):
            # A[seqpos, seqpos] = 1
            if inps[row, seqpos, 1] == 0:  # open
                stack.append(seqpos)
                opened_so_far.append(seqpos)
            elif inps[row, seqpos, 1] == 1:
                openpos = stack.pop()
                power = 1
                A[openpos, seqpos] = power
                A[seqpos, openpos] = power

        As.append(A)
    return np.array(As)


# additional features

def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"{bpps_path}/{mol_id}.npy").max(axis=1))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"{bpps_path}/{mol_id}.npy").sum(axis=1))
    return bpps_arr


def read_bpps_nb(df):
    # normalized non-zero number
    # from https://www.kaggle.com/symyksr/openvaccine-deepergcn
    print(df.shape)
    bpps_nb_mean = 0.077522  # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914  # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"{bpps_path}/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr


def get_inputs(df):
    df['bpps_sum'] = read_bpps_sum(df)
    df['bpps_max'] = read_bpps_max(df)

    unp_df = []

    for id in df.id.values:
        unp_df.append(np.load(bpps_path + '/' + id + '.npy'))
    unp_df = np.array(unp_df)

    return df, unp_df


def load_models(seqlen, n_models=5):
    models = []
    type_ = 0
    for type_ in [0,1]:
        for cv in range(n_models):
            # Load model.
            model_base = build_base_model(seq_len=seqlen, pred_len=seqlen, type=type_)
            model = build_model(model_base, seq_len=seqlen, pred_len=seqlen, type=type_)
            model.load_weights(f'{model_path}/{str(type_)}_modelGRU_LSTM1_cv{cv}.h5')
            print(f"Model loaded: {'#' * 5} Type {type_} Fold {cv} {'#' * 5}")
            models.append(model)
    return models


def get_prediction(data, models):
    holdout_preds = []
    type_ = 0

    # Preprocessing.
    train, unp_train = get_inputs(data)

    # Prepraing inputs for the model.
    x_trn, x_ohe_trn = preprocess_inputs(train)
    train_As = get_adjacency_matrix(x_trn[:, :, :3])
    train_As = np.concatenate(
        [unp_train[:, :train_As.shape[1], :train_As.shape[1], None] ** 50, train_As[:, :, :, None]], axis=3)

    for model in models:
        # Get prediction.
        oof_preds = model.predict([x_trn, x_ohe_trn, train_As])
        print(f"Prediction shape: {oof_preds.shape}")
        # print(tf.reduce_mean(mcrmse(y_trn, oof_preds[:,:y_trn.shape[1],:]))) #calculate score based on given true target length.

        holdout_preds.append(oof_preds)
        gc.collect()
    holdout_preds = np.mean(holdout_preds, axis=0)  # Averaging 5 folds' predictions.
    return holdout_preds

train_json_path = "stanford-covid-vaccine/train.json"
bpps_path = "stanford-covid-vaccine/bpps"
model_path =  "models"




def get_prediction_df_dict(data,models_dict):

    final_holdouts_df = pd.DataFrame()
    seqlen_list = list(models_dict.keys())
    for seqlen in seqlen_list:
        sub_data = data[data["sequence_length"] == seqlen].copy().reset_index(drop=True)
        val_preds = get_prediction(data=sub_data, models=models_dict[seqlen])
        off_preds_ls = []
        for i, uid in enumerate(sub_data.id):
            single_pred = val_preds[i]
            single_df = pd.DataFrame(single_pred, columns=pred_cols)
            single_df['id'] = uid
            single_df['seqpos'] = [x for x in range(single_df.shape[0])]
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
            #single_df['SN_filter'] = train[train['id'] == uid].SN_filter.values[0]
            off_preds_ls.append(single_df)

        holdouts_df = pd.concat(off_preds_ls)
        columns_order = ["id"] + [col for col in holdouts_df.columns if col not in ["id"]]
        holdouts_df = holdouts_df[columns_order]
        final_holdouts_df = pd.concat([final_holdouts_df,holdouts_df],axis=0)
    print(f"Prediction dataframe shape: {final_holdouts_df.shape}")
    return final_holdouts_df

def bpps_check(df):
    path_list = []
    for file in os.listdir(bpps_path):
        path_list.append(os.path.join(file).split(".")[0])
    bpps_list_to_generate = [id for id in df.id.values if id not in path_list]  # intersection list
    return bpps_list_to_generate

# you can test the code here.
#train = pd.read_json(train_json_path, lines=True)
#train["sequence_length"] = train["sequence"].apply(lambda x: len(x))
#minimum_required_features = ["id","sequence","structure","predicted_loop_type","sequence_length"]
#get_preds_data = train.loc[:100].copy()

#seqlen = len(get_preds_data.loc[0,"sequence"])
#models_dict = {}
#models = load_models(seqlen,n_models = 1)
#models_dict[seqlen] = models

#holdouts_df = get_prediction_df_dict(data=get_preds_data.loc[:,minimum_required_features],models_dict=models_dict)

#holdouts_df.head()
