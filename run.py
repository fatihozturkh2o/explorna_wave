# Form / Frame
# Use a frame component in a form card to display HTML content inline.
# ---
import sys
sys.path.append('draw_rna_pkg/')
from h2o_wave import Q, listen, ui
import matplotlib.pyplot as plt
from ipynb.draw import draw_struct
import os
import io
import base64
import matplotlib.image as mpimg
import zipfile
import time



os.environ["ARNIEFILE"] = f"arnie.conf"


from rna_analysis import *
from model_prediction import load_models,bpps_path,get_prediction_df_dict,bpps_check






# #for cmaps check out https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
target_columns = [ 'reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C']

logo_file = "files/logo.png"
image_1_path = "files/image_1_path.png"
image_2_path = "files/image_2_path.png"
image_3_path = "files/image_3_path.png"
image_4_path = "files/image_4_path.png"

all_pages = ['nav','home', 'upload', 'example', 'infoboard1', 'image0', 'image1', 'image2','data_info','stn_hist',
                 'image3', 'image4', 'image5', 'description', 'text','text2', 'error', 'progress', 'selection_card', 'plot',
                 "plot_seqpos","plot_count",'plot_features0','plot_features1','plot_features2','plot_features3','text_sub','predict_tab']


image_size = 4
number_of_plots_in_a_row = 3
plot_height = 5
plot_width = 3

data_display_max_nrows = 10

def split_text(str_):
    return str_.split(" ")


def compress(file_names):


    path = ""

    # Select the compression mode ZIP_DEFLATED for compression
    # or zipfile.ZIP_STORED to just store the file
    compression = zipfile.ZIP_DEFLATED

    # create the zip file first parameter path/name, second mode
    zf = zipfile.ZipFile("temp/plots.zip", mode="w")
    try:
        for file_name in file_names:
            # Add file to the zip file
            # first parameter file to zip, second filename in zip
            zf.write(path + file_name, file_name, compress_type=compression)

    except:
        print("ERROR: No files to zip")
    finally:
        # Don't forget to close the file!
        zf.close()




def get_cell_axis(cell_multiplier = 10):
    cell_size_x = 72 * cell_multiplier
    cell_size_y = 72 * cell_multiplier
    fig, ax = plt.subplots(1, 1, figsize=(cell_size_x / 72, cell_size_y / 72))
    del fig
    return ax
def get_random_id(length=8):
    letters = [c for c in "abcdefghijklmnopqrstuwxyz0123456789"]
    result_str = ''.join(np.random.choice(letters) for i in range(length))
    result_str = "id_custom" + result_str
    return result_str

def get_random_sample(df,seq_col,struct_col,target_columns=target_columns):
    sample_ = np.array(int(np.random.uniform(len(df))))
    seq = "".join(df.loc[sample_][seq_col])
    struct = "".join(df.loc[sample_][struct_col])
    bp_matrix = bpps(seq)
    unpaired_probs = 1 - np.sum(bp_matrix, axis=0)
    df_sample = df.loc[sample_,target_columns]
    df_sample["bpps"] = unpaired_probs
    sample_id = df.loc[sample_, "id"]
    return seq, struct,df_sample,sample_id

def get_sample(df,id,target_columns = target_columns):
    sample_ = df[df["id"]==id].index[0]
    seq = "".join(df.loc[sample_]["sequence"])
    struct = "".join(df.loc[sample_]["structure"])
    bp_matrix = bpps(seq)
    unpaired_probs = 1 - np.sum(bp_matrix, axis=0)
    df_sample = df.loc[sample_,target_columns]
    df_sample["bpps"] = unpaired_probs
    sample_id = df.loc[sample_, "id"]
    return seq, struct,df_sample,sample_id

def get_image(file_name = "image1"):
    plt.figure(figsize=(image_size, image_size))
    buf = io.BytesIO()

    img = mpimg.imread(f'temp/{file_name}.png')
    plt.imshow(img)
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode('utf-8')

    return image

def display_error(error_type):
    items = []
    if error_type == "upload":
        items = [ui.message_bar(type='error', text='**Selected file was not in the expected format.**'),
                    ui.button(name='upload_data', label='Upload again', primary=True),
                    ui.separator()]
    elif error_type == "misentered_info":
        items = [ui.message_bar(type='error', text='**Either sequence or structure info not entered.**'),
                ui.button(name='custom_back', label='Try again', primary=True),
                ui.separator()]
    elif error_type == "wrong_columns":
        items = [
                ui.message_bar(type='error', text='**Either sequence or structure column not selected properly**'),
                ui.button(name='restart', label='Try again', primary=True),
                ui.separator()
            ]
    elif error_type == "misentered_ids":
        items = [
                ui.message_bar(type='error', text='**IDs not entered properly**'),
                ui.button(name='restart', label='Try again', primary=True),
                ui.separator()
            ]
    elif error_type == "arnie_not_selected":
        items = [
            ui.message_bar(type='error', text='**Either feature or package not selected properly.**'),
            ui.button(name='arnie_back', label='Try again', primary=True),
            ui.separator()
        ]
    return items

def make_ui_table(df, n_rows, name = "head_of_table"):
    """Creates a ui.table object from a csv file"""

    n_rows = min(n_rows, df.shape[0])

    table = ui.table(
        name=name,
        columns=[ui.table_column(name=str(x), label=str(x), sortable=True) for x in df.columns.values],
        rows=[ui.table_row(name=str(i), cells=[str(df[col].values[i]) for col in df.columns.values])
              for i in range(n_rows)]
    )
    return table

async def delete_pages(q: Q,keep_nav=False):
    # There does not seem to be a way to know what pages are in q.page
    page_list = ['nav','home', 'upload', 'example', 'infoboard1','data_info','stn_hist',
                 'description', 'text','text2', 'error', 'progress', 'selection_card', 'plot','data_view',
                 "plot_seqpos","plot_count",'plot_features0','plot_features1','plot_features2','plot_features3','text_sub','predict_tab'] \
                + list(set(q.client.all_pages))
    if keep_nav:
        page_list.remove("nav")

    for page in page_list:
        try:
            del q.page[page]
        except:
            pass

async def display_nav(q: Q):
    q.page['nav'] = ui.tab_card(
        box='3 1 9 1',
        items=[
            ui.tab(name='#home', label='Home'),
            ui.tab(name='#plotrnas', label='Plot RNAs'),
            ui.tab(name='#predict', label='Predict Data'),
            ui.tab(name='#analysis', label='Data Analysis'),
            ui.tab(name='#arniefeatures', label='ARNIE Features'),
            ui.tab(name='#customsample', label='Custom Sample'),
            ui.tab(name='#uploaddataset', label='Upload Dataset')

        ],
        link=False
    )

    q.page['header2'] = ui.header_card(
        box='1 1 2 1',
        title='E x p l o R N A',
        subtitle='',
        icon='ExploreData',
        icon_color='$orange',
    )

    q.page['logo'] = ui.markup_card(
        box='12 1 1 1',
        title='',
        content="""<p style='text-align:center; vertical-align: middle; display: table-cell; width: 134px;'>"""
                """<a href='https://www.h2o.ai/products/h2o-wave/'> <img src='""" + q.app.logo_url + """' height='50px' width='50px'> </a> </p>""" )

async def progress_page(q: Q, message='\nJust a second...'):
    q.page['progress'] = ui.form_card(box='4 5 6 2',
                                      items=[ui.progress(label=message)])
    await q.page.save()
    del q.page['progress']



async def display_file_upload(q):
    print("location: display_file_upload")

    q.page['description'] = ui.form_card(box='1 2 3 10',
        items=[ui.text_m('\n You can upload a local dataset to run ExploRNA.'),
               ui.file_upload(name='user_files', label='Upload', multiple=False)])

    if q.client.train is not None:
        data_items = [ui.text_m(f'Loaded file "{q.client.file_name}" has '
                                f'**{q.client.train.shape[0]}** rows and **{q.client.train.shape[1]}** features.\n\n'),
                      make_ui_table(q.client.train, data_display_max_nrows)]
        q.page['data_view'] = ui.form_card(box='4 2 9 7', items=data_items)


async def plotrnas(q):
    colormap_choices = []
    colormap_columns = q.client.target_columns + ["bpps"]


    for jj in colormap_columns:
        choice = ui.choice(jj, jj)
        colormap_choices.append(choice)

    info_text = '''
As plotting tool, we use draw_rna developed by the Das Lab, Eterna Players, M. Wu, H. Wayment-Steele.
 You can click [here](https://github.com/DasLab/draw_rna) to open the github repo.
'''
    ui_list = [
        ui.text_m('In this section, you can generate RNA plots by using the uploaded dataset. You can either type id'
                  ' or randomly select rows to be plotted.'),
        ui.text_m('After generating the plots, you can download them to your local.'),
        ui.text_s('Example IDs: id_001f94081,id_0049f53ba,id_006f36f57'),
        ui.dropdown(name='colormap', label='Select colormap for plots', value=None, required=False,
                    choices=colormap_choices),
        ui.textbox(name='id_textbox', label='ID', placeholder='You can enter multiple ids. Separate with comma.'),
        ui.button(name='plot_id', label='Plot', primary=True),
        ui.slider(name='plotnumber_slider', label='Select number of random plots', min=1, max=6, step=1, value=3),
        ui.buttons([ui.button(name='random_sample', label='Random Plot', primary=False)]),
        ui.text_m(info_text),
        ]
    q.page['text'] = ui.form_card(box='1 2 3 10', items=ui_list)


async def arnie_features(q):
    feature_choices = []
    for jj in ["bpps","mfe","unpaired_proba"]:
        choice = ui.choice(jj, jj)
        feature_choices.append(choice)

    package_choices = []
    for jj in ["vienna_2"]:
        choice = ui.choice(jj, jj)
        package_choices.append(choice)


    ui_list = [ui.text_m(f'In this section, you can generate RNA related features provided by ARNIE package.'
                         f' ARNIE is a Python API to compute RNA energetics and do structure prediction across multiple secondary structure packages. '
                         f' You can reach the ARNIE github repo by clicking [here](https://github.com/DasLab/arnie).'
                         f' Currently, ExploRNA includes only some of the packages in ARNIE.'),
               ui.text_m(f'You can download the features to your local after they are generated.'),
        ui.dropdown(name='feature_generate', label='Select feature to generate', value=None, required=False,
                    choices=feature_choices),
        ui.dropdown(name='package_generate', label='Select package', value=None, required=False,
                           choices=package_choices),
        ui.buttons([ui.button(name='select_arnie', label='Select', primary=False)]),
    ]
    q.page['text'] = ui.form_card( box='1 2 3 10',items=ui_list )

    if q.args.select_arnie:
        print("location: select_arnie")
        if q.args.feature_generate is not None and q.args.package_generate is not None:
            start_time_main = time.time()
            q.page['progress'] = ui.form_card(box='4 2 3 3',
                                              items=[ui.progress(label='\nJust a second...')])
            await q.page.save()

            s = arnie_feature(df=q.client.train, feature=q.args.feature_generate, package= q.args.package_generate, sequence_column="sequence")
            path_name = q.args.feature_generate + '_' + q.args.package_generate + '_feature'
            s.to_csv(f'temp/{path_name}.csv', index=False)
            q.client.arnie_feature, = await q.site.upload([f'temp/{path_name}.csv'])

            data_text = '''=
Click [here]({{arnie_feature_path}}) to download the generated {{arnie_feature}} feature.
    '''
            del q.page['progress']
            q.page['text2'] = ui.markdown_card(
            box='4 2 3 3',
            title='',
            content=data_text,
            data=dict(arnie_feature_path=q.client.arnie_feature,
                      arnie_feature=q.args.feature_generate))

            elapsed_time = time.time() - start_time_main
            print(f"minutes passed for getting feature: {round(elapsed_time / 60, 2)}")
        else:
            await delete_pages(q,keep_nav=True)
            q.page['error'] = ui.form_card(
                box='1 2 6 2',
                items=display_error(error_type="arnie_not_selected"))


async def data_analysis(q):
    print("location: dataanalysis")
    if q.client.data_analysis is None:

        unseq_df = get_unseq_data(q.client.train,target_columns=q.client.target_columns)
        print(unseq_df.shape)

        # Sequence Position based plot
        q.client.html_pos = position_based_plot(unseq_df.copy(),q.client.target_columns)
        q.client.html_count = sequence_length_counts(q.client.train, seq_column="sequence")
        q.client.html_features_list = []
        for i, f_ in enumerate(["structure", "sequence", "predicted_loop_type"]):
            q.client.html_features_list.append(feature_base_target_means(unseq_df.copy(), q.client.target_columns, features=[f_]))

        q.client.data_analysis = True

        if "signal_to_noise" in q.client.train.columns:
            hist, edges = np.histogram(q.client.train["signal_to_noise"].values, density=False, bins=30)
            q.client.html_stnhist = make_hist_plot("Signal to Noise Ratio Histogram", hist, edges)

    q.page['plot_count'] = ui.frame_card(
        box='6 2 5 4',
        title='Number of samples based on sequence length',
        content=q.client.html_count
    )

    if q.client.target_columns:
        q.page['plot_seqpos'] = ui.frame_card(
            box='1 2 5 8',
            title='Average target value plots for each sequence position.',
            content=q.client.html_pos
        )


        for i, html_features in enumerate(q.client.html_features_list):
            q.page[f'plot_features{i}'] = ui.frame_card(
                box=f'6 {6 + i*4} 5 4',
                title='Average target values based on unique feature values',
                content=html_features
            )

    if "signal_to_noise" in q.client.train.columns:
        q.page['stn_hist'] = ui.frame_card(
            box='1 10 5 8',
            title='',
            content=q.client.html_stnhist
        )

async def model_predict(q):

    # Loading models
    start_time_main = time.time()

    if q.client.models_loaded is None:
        await progress_page(q, message="Loading models...")
        # Here we get model for each unique sequence length. This is essential because both preprocessing and
        # loading models part will be based on sequence lengths.
        seqlen_list = list(q.client.train["sequence_length"].unique())
        q.client.models_dict = {}
        for seqlen in seqlen_list:
            print(seqlen)
            q.client.models_dict[seqlen] = load_models(seqlen, n_models=q.client.number_of_models)
        q.client.models_loaded = True



    # TODO: Here I need to add a check for bpps matrices.
    if q.client.bpps_list_to_generate is None:
        await progress_page(q, message="Checking missing bpps matrices...")
        q.client.bpps_list_to_generate = bpps_check(q.client.train)

    if len(q.client.bpps_list_to_generate) > 0:
        print("location: bpps_list_to_generate")
        print(len(q.client.bpps_list_to_generate))
        await progress_page(q, message="Creating bpps matrices...")
        counter = 1
        start_time = time.time()
        for id in q.client.bpps_list_to_generate:
            sample_ = q.client.train[q.client.train["id"] == id].index[0]
            seq = "".join(q.client.train.loc[sample_]["sequence"])
            bp_matrix = bpps(seq)
            np.save(bpps_path + f'/{id}.npy', bp_matrix)  # This saved file will be used during model prediction.
            if counter%500 == 0:
                print(counter)
            counter += 1

        q.client.bpps_list_to_generate = None

        elapsed_time = time.time() - start_time
        print(f"time passed during bpps matrix creation: {round(elapsed_time, 2)}")


    # Getting predictions
    if q.client.predictions is None:
        minimum_required_features = ["id","sequence","structure","predicted_loop_type","sequence_length"]
        await progress_page(q,message="Getting predictions...")
        q.client.predictions = get_prediction_df_dict(data=q.client.train.loc[:, minimum_required_features], models_dict=q.client.models_dict)
        q.client.predictions.to_csv('temp/predictions.csv', index=False)
        q.client.predictions_path, = await q.site.upload(['temp/predictions.csv'])

    elapsed_time = time.time() - start_time_main
    print(f"minutes passed for getting predictions: {round(elapsed_time/60, 2)}")
    data_text = '''=
Click [here]({{predictions}}) to download the predictions.
'''
    q.page['text'] = ui.markdown_card(
            box='4 9 4 1',
            title='',
            content=data_text,
            data=dict(predictions=q.client.predictions_path)
        )

async def predict_tab(q):
    q.page['predict_tab'] = ui.form_card(box='1 2 3 10', items=[
        ui.text_m(f'In this section, you can generate predictions for the whole uploaded dataset.'),
        ui.text_m(f'Predictions will be generated for following targets: {target_columns}'),
        ui.text_m(f'IMPORTANT: To be able to run Predict succesfully you need bpps matrices for each id in the data.'
                  f' It is recommended to put them as npy files in following path: /stanford-covid-vaccine/bpps/'
                  f' In case of missing file, the app will automatically generate bpps matrices by using bpps function and vienna_2 package from arnie API.'),
        ui.slider(name='modelnumber_slider', label='Select number of model predictions for final blending.', min=1, max=5, step=1, value=1),
        ui.button(name='dataprediction', label='Predict', primary=True),
    ])


async def home(q):
    home_text = f'''
The ExploRNA app has been developed based on the public dataset shared for [OpenVaccine kaggle competition](https://www.kaggle.com/c/stanford-covid-vaccine). The models used for predictions were developed by Anthony and Fatih who placed in 11th position under the team name of "Social Distancing Please" in the same competition. You can read the simplified model architecture [here](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189571).

Purpose of the competition was predicting multiple targets that were representing likely degradation rates for each base in a given RNA molecule. This was an important task because being able to find out where in the backbone of a given RNA is most prone to being affected can play an important role for the production of Covid19 vaccines. For this study of Stanford University, they focus on directly injecting mRNAs to human body instead of traditional vaccines and by doing so they aim human body to produce antibodies against Covid19 virus. However, RNAs have quite sensitive structures in their nature and they can easily deteriorate outside the cell. Hence, it is helpful to find out which parts in a given sequence degrade by leveraging machine learning algorithms.

![Plot]({q.app.home_image_1_url})

Image source: https://www.kaggle.com/c/stanford-covid-vaccine

Throughout the app, you will be able to plot RNAs, generate predictions for degradation targets, generate some ARNIE features and also analyse the data via plots. ARNIE is a Python API developed by DasLab to compute RNA energetics and do structure prediction across multiple secondary structure packages. You can reach the github repo by clicking [here](https://github.com/DasLab/arnie). In Custom Sample section you will not need the uploaded dataset. You can just enter any sequence, structure and predicted loop type info and get both RNA plot and predictions.

_____________________![Plot]({q.app.home_image_2_url}) ![Plot]({q.app.home_image_3_url}) ![Plot]({q.app.home_image_4_url})

 '''
    q.page['home'] = ui.form_card(box='1 2 12 10',
        items=[
               ui.text_m(home_text)
               ])

async def display_data_parameters_page(q,random_sample_disabled=True):

    # Read the file
    print("location: data parameters.")

    choices_target = []
    for jj in range(len(q.client.fs_columns)):
        choice = ui.choice(q.client.fs_columns[jj], q.client.fs_columns[jj])
        choices_target.append(choice)



    choices_feat = []
    for jj in q.client.fs_columns:
            choice = ui.choice(jj, jj)
            choices_feat.append(choice)

    ui_list_intro = [ui.text_l('**Dataset**'),
                    ui.text_m(f'Loaded file "{q.client.file_name}" has '
                                 f'**{q.client.train.shape[0]}** rows and **{q.client.train.shape[1]}** features.\n\n'),
                    ui.text_m(
                         f'\nBy selecting sequence and structure columns, you can generate random RNA plots from the data.'),
                    ui.separator("Specify Columns"),
                    ui.dropdown(name='id_dd', label='Select ID column', value=None, required=True,
                                 choices=choices_target),
                    ui.dropdown(name='sequence_dd', label='Select sequence column', value=None, required=True,
                                 choices=choices_target),
                    ui.dropdown(name='structure_dd', label='Select structure column', value=None, required=True,
                                 choices=choices_target),
                    ui.dropdown(name='predictedloop_dd', label='Select predicted loop type column', value=None, required=True,
                                 choices=choices_target),
                    ui.checklist(name='checklist', label='Select target columns', choices=choices_feat),
                    ui.button(name='select', label='Select', primary=True),

                     ]

    ui_list_custom = [
        ui.text_m(f'In this section, you can directly type sequence,structure and predicted_loop_type infos. '),
        ui.text_m(f'Then you can both generate RNA plot and model predictions for following targets: {target_columns}.'),
        ui.textbox(name='sequence_textbox', label='Sequence', value="GGAAAAGCUCUAAUAACAGGAGACUAGGACUACGUAUUUCUAGGUAACUGGAAUAACCCAUACCAGCAGUUAGAGUUCGCUCUAACAAAAGAAACAACAACAACAAC"),
        ui.textbox(name='structure_textbox', label='Structure', value=".....((((((.......)))).)).((.....((..((((((....))))))..)).....))....(((((((....)))))))....................."),
        ui.textbox(name='predictedlooptype_textbox', label='Predicted Loop Type',value="EEEEESSSSSSHHHHHHHSSSSBSSXSSIIIIISSIISSSSSSHHHHSSSSSSIISSIIIIISSXXXXSSSSSSSHHHHSSSSSSSEEEEEEEEEEEEEEEEEEEEE"),
        ui.button(name='show_inputs', label='Generate', primary=True),
        ui.text_s(f'You can enter in following formats: '),
        ui.text_s(f'Sequence characters: A,U,C,G'),
        ui.text_s(f'Structure characters: .,(,)'),
        ui.text_s(f'\nLoop Type characters: B,E,H,I,M,S,X')

    ]

    # Add navigation buttons
    #q.page['text'] = ui.form_card(box='1 2 5 10', items=ui_list_intro + ui_list)
    print(q.args["#"])
    if q.client.activetab == "dataproperties":
            await delete_pages(q,keep_nav=True)
            q.page['text'] = ui.form_card( box='1 2 3 10',items=ui_list_intro )
    elif q.client.activetab == "plotrnas":
            await delete_pages(q,keep_nav=True)
            await plotrnas(q)
    elif q.client.activetab == "customsample":
            await delete_pages(q,keep_nav=True)
            q.page['text'] = ui.form_card( box='1 2 3 10',items=ui_list_custom )
    elif q.client.activetab == "analysis":
            await delete_pages(q,keep_nav=True)
            await progress_page(q)
            await data_analysis(q)
    elif q.client.activetab == "arniefeatures":
            await delete_pages(q,keep_nav=True)
            await arnie_features(q)
    elif q.client.activetab == "predict":
            await delete_pages(q,keep_nav=True)
            await predict_tab(q)
    elif q.client.activetab == "home":
            await delete_pages(q,keep_nav=True)
            await home(q)
    elif q.client.activetab == "uploaddataset":
            await delete_pages(q,keep_nav=True)
            await display_file_upload(q)





async def main(q: Q):
    # Upload the logo & images
    if q.client.all_pages is None:
        q.client.all_pages = []

    if not q.app.logo_url:
        q.app.logo_url, = await q.site.upload([logo_file])

    if not q.app.home_image_1:
        q.app.home_image_1_url, = await q.site.upload([image_1_path])

    if not q.app.home_image_2:
        q.app.home_image_2_url, = await q.site.upload([image_2_path])

    if not q.app.home_image_3:
        q.app.home_image_3_url, = await q.site.upload([image_3_path])

    if not q.app.home_image_4:
        q.app.home_image_4_url, = await q.site.upload([image_4_path])

    if q.args.user_files:
        #await delete_pages(q)
        try:
            print('location: upload data')
            # Make the file available locally and store file path in client context
            q.client.local_path = await q.site.download(q.args.user_files[0], '.')
            q.client.link_to_file = q.args.user_files[0]

            q.client.train = pd.read_json(q.client.local_path, lines=True)
            q.client.fs_columns = list(q.client.train.columns.values.tolist())
            q.client.file_name = os.path.split(q.client.local_path)[1]
            q.client.target_columns = [col for col in target_columns if col in q.client.train.columns]
            q.client.train["sequence_length"] = q.client.train["sequence"].apply(lambda seq: len(seq))

            # Following lines are needed to re-analyze and re-predict with the recently uploaded dataset.
            q.client.data_analysis = None
            q.client.predictions = None
            q.client.models = None
            q.client.models_loaded = None
            q.client.bpps_list_to_generate = None

            print(f"data shape: {q.client.train.shape}")
            data_items = [ui.text_m(f'Loaded file "{q.client.file_name}" has '
                                    f'**{q.client.train.shape[0]}** rows and **{q.client.train.shape[1]}** features.\n\n'),
                          make_ui_table(q.client.train, data_display_max_nrows)]
            q.page['data_view'] = ui.form_card(box='4 2 9 7', items=data_items)



        except:
            await delete_pages(q, keep_nav=True)
            q.page['error'] = ui.form_card(box='1 2 6 2',
                items=display_error(error_type="upload"))

    elif q.client.local_path is None:
        q.client.local_path = "stanford-covid-vaccine/train_small.json"
        q.client.train = pd.read_json(q.client.local_path, lines=True)
        q.client.fs_columns = list(q.client.train.columns.values.tolist())
        q.client.file_name = os.path.split(q.client.local_path)[1]
        q.client.activetab = "home"
        q.client.target_columns = [col for col in target_columns if col in q.client.train.columns]
        q.client.train["sequence_length"] = q.client.train["sequence"].apply(lambda seq: len(seq))


        await display_data_parameters_page(q)
        await display_nav(q)
        #await display_file_upload(q)

    elif q.args.upload_data:
        await delete_pages(q,keep_nav=True)
        q.client.activetab = "uploaddataset"
        await display_data_parameters_page(q)

    elif q.args.show_inputs:
        await delete_pages(q,keep_nav=True)
        await progress_page(q)
        await show_inputs(q)

    elif q.args.random_sample:
        await delete_pages(q,keep_nav=True)
        await progress_page(q)
        await random_sample(q)

    elif q.args.plot_id:
        print("location: plot_id")
        print(q.args.id_textbox)
        q.client.idlist = q.args.id_textbox.replace(" ","").split(",")
        await delete_pages(q,keep_nav=True)
        await progress_page(q)
        await plot_id(q)





    elif q.args.select:
        q.client.id_column = q.args.id_dd
        q.client.sequence_column = q.args.sequence_dd
        q.client.structure_column = q.args.structure_dd
        q.client.predictedloop_column = q.args.predictedloop_dd
        q.client.checklist = q.args.checklist
        q.client.colormap = q.args.colormap

        print("location: select")
        q.page['selection_card'] = ui.form_card(box='4 2 3 3',
        items=[ui.text_l('**Selected columns **'),
        ui.text_m(f'\nID column: {q.client.id_column}'),
        ui.text_m(f'\nSequece column: {q.client.sequence_column}'),
        ui.text_m(f'\nStructure column: {q.client.structure_column}'),
        ui.text_m(f'\nPredicted loop type column: {q.client.predictedloop_column}'),
        ui.text_m(f'\nTarget columns: {q.client.checklist}')])

        #await display_data_parameters_page(q,random_sample_disabled=False)
        await display_nav(q)

    elif q.args.restart:
        await delete_pages(q, keep_nav=True)
        # Make sure that first active tab is dataproperties.
        q.client.activetab = "plotrnas"
        await display_data_parameters_page(q)

    elif q.args.custom_back:
        # Make sure that first active tab is dataproperties.
        await delete_pages(q,keep_nav=True)
        q.client.activetab = "customsample"
        await display_data_parameters_page(q)

    elif q.args.arnie_back:
        # Make sure that first active tab is dataproperties.
        await delete_pages(q,keep_nav=True)
        q.client.activetab = "arniefeatures"
        await display_data_parameters_page(q)



    elif q.args.customprediction:
        # Loading models

        await progress_page(q, message="Loading models...")
        seqlen_list = list(q.client.customdata["sequence_length"].unique())
        q.client.custommodels_dict = {}
        for seqlen in seqlen_list:
            q.client.custommodels_dict[seqlen] = load_models(seqlen, n_models=q.args.modelnumber_slider)


        # Getting predictions
        minimum_required_features = ["id", "sequence", "structure", "predicted_loop_type","sequence_length"]
        await progress_page(q, message="Getting predictions...")
        q.client.custompredictions = get_prediction_df_dict(data=q.client.customdata.loc[:, minimum_required_features],
                                                 models_dict=q.client.custommodels_dict)
        q.client.custompredictions.to_csv('temp/custompredictions.csv', index=False)
        q.client.custompredictions_path, = await q.site.upload(['temp/custompredictions.csv'])

        q.client.customdata.to_csv('temp/customdata.csv', index=False)
        q.client.customdata_path, = await q.site.upload(['temp/customdata.csv'])

        data_text = '''=
Click [here]({{predictions}}) to download the custom predictions. You can also download your custom data by clicking [here]({{data}}).
        '''
        q.page['text'] = ui.markdown_card(
            box='4 7 3 2',
            title='',
            content=data_text,
            data=dict(predictions=q.client.custompredictions_path, data=q.client.customdata_path)
        )

        q.client.custom_html_pos = position_based_plot(q.client.custompredictions, target_columns)
        q.page['plot_seqpos'] = ui.frame_card(
            box='7 2 5 8',
            title='Prediction value plots for each sequence position.',
            content=q.client.custom_html_pos
        )

        #This part is for controlling active tab page.
    elif q.args.dataprediction:
        q.client.number_of_models = q.args.modelnumber_slider
        await model_predict(q)
        await predict_tab(q)

        q.client.data_predictions_html_pos = position_based_plot(q.client.predictions, target_columns, size=500)
        q.page['plot_seqpos'] = ui.frame_card(
            box='4 2 4 7',
            title='Average prediction value plots for each sequence position.',
            content=q.client.data_predictions_html_pos
        )

    elif q.args["#"]:
        q.client.activetab = q.args["#"]
        await display_data_parameters_page(q)

    else:
        print("location: else")
        await display_data_parameters_page(q)

    await q.page.save()


async def show_inputs(q: Q):
    if q.args.sequence_textbox and q.args.structure_textbox:
        seq = q.args.sequence_textbox
        struct = q.args.structure_textbox
        pltype = q.args.predictedlooptype_textbox
        bp_matrix = bpps(seq)
        custom_id = get_random_id(length=8)
        seqlen = len(seq)
        np.save(bpps_path+f'/{custom_id}.npy',bp_matrix) # This saved file will be used during model prediction.

        custom_dict = {"id": custom_id,
         "sequence": [seq],
         "structure": [struct],
         "predicted_loop_type":[pltype],
         "sequence_length": seqlen
         }
        q.client.customdata = pd.DataFrame(custom_dict)


        # Generate image
        ax = get_cell_axis(cell_multiplier = 10)
        draw_struct(seq, struct, c=None, ax=None,file_name=custom_id)
        image = get_image(file_name=custom_id)

        image_path_list = [f"temp/{custom_id}.png"]
        compress(image_path_list)
        # Deleting previously generated plots after compressing them.
        for path in image_path_list:
            os.remove(path)

        q.client.plots_path, = await q.site.upload([f'temp/plots.zip'])

        q.page['image1'] = ui.image_card(
            box=f'4 2 3 5',
            title=f'RNA Visualization for ID = {custom_id}:',
            type='png',
            image=image)

        q.client.all_pages.append("image1")

        q.page['infoboard1'] = ui.form_card(box='1 2 3 10', items=[])
        q.page['infoboard1'].items = [
            ui.text(f'Random ID = {custom_id}'),
            ui.text(f'Sequence = {seq}'),
            ui.text(f'Structure = {struct}'),
            ui.text(f'Predicted Loop Type = {pltype}'),
            ui.button(name='custom_back', label='Back', primary=True),
            ui.slider(name='modelnumber_slider', label='Select number of model predictions for final blending.', min=1, max=5, step=1, value=1),
            ui.button(name='customprediction', label='Get Prediction', primary=False),
        ]

        data_text = '''=
Click [here]({{plots_path}}) to download the generated plot.
                        '''
        q.page['text2'] = ui.markdown_card(
            box='1 10 3 1',
            title='',
            content=data_text,
            data=dict(plots_path=q.client.plots_path)
        )
    else:
        q.page['error'] = ui.form_card(
            box='1 2 6 2',
            items=display_error(error_type="misentered_info"))

async def plot_id(q: Q):
    print("Location: Plot ID function")

    try:
        image_list = []
        image_path_list = []
        for id in q.client.idlist:
            seq, struct, df_sample,sample_id = get_sample(q.client.train, id, target_columns=q.client.target_columns)
            ax = get_cell_axis(cell_multiplier=10)
            image_path_list.append(f"temp/{sample_id}.png")

            if q.args.colormap is None:
                draw_struct(seq, struct, ax=None,file_name=sample_id)
            else:
                # Following part: Since the data's target columns may not have values for each element in sequences
                # I'm filling missing values with 0s if there are any.
                colormap_list = list(df_sample[q.args.colormap])
                colormap_list = colormap_list + list(np.zeros(len(seq) - len(colormap_list)))
                draw_struct(seq, struct, c=colormap_list, ax=None,file_name=sample_id)
            image = get_image(file_name=sample_id)
            image_list.append(image)

        for i, image in enumerate(image_list):
            box_location = f'{4 + plot_width * (i % number_of_plots_in_a_row)} {2 + plot_height * int(i / number_of_plots_in_a_row)} {plot_width} {plot_height}'

            q.page[f'image{i}'] = ui.image_card(
                box=box_location,
                title=f'RNA Visualization for ID = {q.client.idlist[i]}:',
                type='png',
                image=image)

            q.client.all_pages.append(f'image{i}') # Update the pages.

        image_path_list = list(set(image_path_list)) # Since same ids can be entered multiple times, we need to only use unique paths.
        compress(image_path_list)
        # Deleting previously generated plots after compressing them.
        for path in image_path_list:
            os.remove(path)

        await plotrnas(q)
        q.client.plots_path, = await q.site.upload([f'temp/plots.zip'])
        data_text = '''=
Click [here]({{plots_path}}) to download the generated plots.
        '''
        del q.page['progress']
        q.page['text2'] = ui.markdown_card(
            box='1 10 3 1',
            title='',
            content=data_text,
            data=dict(plots_path=q.client.plots_path)
        )


    except:
        q.page['error'] = ui.form_card(
            box='1 2 6 2',
            items=display_error(error_type="misentered_ids"))




async def random_sample(q: Q):
    print("Location: Random Sample")

    try:

        image_list = []
        image_path_list = []
        sample_id_list = []
        for i in range(q.args.plotnumber_slider):
            seq, struct, df_sample,sample_id = get_random_sample(q.client.train, "sequence", "structure",target_columns=q.client.target_columns)
            image_path_list.append(f"temp/{sample_id}.png")
            sample_id_list.append(sample_id)


            ax = get_cell_axis(cell_multiplier=10)

            if q.args.colormap is None:
                draw_struct(seq, struct, ax=None, file_name=sample_id)
            else:
                # Following part: Since the data's target columns may not have values for each element in sequences
                # I'm filling missing values with 0s if there are any.
                colormap_list = list(df_sample[q.args.colormap])
                colormap_list = colormap_list + list(np.zeros(len(seq) - len(colormap_list)))

                draw_struct(seq, struct, c=colormap_list, ax=None, file_name=sample_id)
            image = get_image(file_name=sample_id)
            image_list.append(image)

        for i, image in enumerate(image_list):
            box_location = f'{4 + 3 * (i % 3)} {2 + 5 * int(i / 3)} 3 5'

            q.page[f'image{i}'] = ui.image_card(
                box=box_location,
                title=f'RNA Visualization for ID = {sample_id_list[i]}:',
                type='png',
                image=image)

            q.client.all_pages.append(f'image{i}')

        image_path_list = list(set(image_path_list))
        compress(image_path_list)
        # Deleting previously generated plots after compressing them.
        for path in image_path_list:
            os.remove(path)

        await plotrnas(q)
        q.client.plots_path, = await q.site.upload([f'temp/plots.zip'])
        data_text = '''=
Click [here]({{plots_path}}) to download the generated plots.
                '''
        del q.page['progress']
        q.page['text2'] = ui.markdown_card(
            box='1 10 3 1',
            title='',
            content=data_text,
            data=dict(plots_path=q.client.plots_path)
        )
    except:
        await delete_pages(q, keep_nav=True)
        q.page['error'] = ui.form_card(
            box='1 2 6 2',
            items=display_error(error_type="wrong_columns"))


if __name__ == '__main__':
    listen('/explorna', main)

