from __future__ import annotations
from os.path import dirname, join
import os
import pickle
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row, Spacer, gridplot
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput, Button, Span
from bokeh.models import TableColumn, DataTable, CustomJS, Circle, MultiLine, Line
from bokeh.events import ButtonClick
from bokeh.transform import linear_cmap, factor_cmap, transform
from bokeh.palettes import RdYlBu
from bokeh.plotting import figure
from bokeh.models.widgets import Tabs, Panel, HTMLTemplateFormatter
from sklearn.datasets import load_boston
from bokeh.models import CustomJS, MultiChoice
from .util import save_pickle, load_pickle, load_yaml, get_dependence_values, decision_value_core, get_importance_list
import yaml
import matplotlib.colors as mc
import matplotlib.pyplot as plt


step_dict = dict()
step_dict['CRIM'] = 0.00001
step_dict['ZN'] = 0.5
step_dict['INDUS'] = 0.01
step_dict['CHAS'] = 1
step_dict['NOX'] = 0.001
step_dict['RM'] = 0.01
step_dict['AGE'] = 0.1
step_dict['DIS'] = 0.0001
step_dict['RAD'] = 1
step_dict['TAX'] = 1
step_dict['PTRATIO'] = 0.1
step_dict['B'] = 0.01
step_dict['LSTAT'] = 0.01


class WhatIfTool:
    def __init__(self, config_path):
        self.config_path = config_path
        self.bck_color = "#2F2F2F"
        self.txt_color = "#D3D3D3"
        self.tools = "pan, wheel_zoom, box_zoom, reset, save, box_select, hover,lasso_select, help"
        self.tooltips = [
            ("Name", "$name"),
            ("ID", "@ID"),
            ("Index", "@index"),
            ("Value", "$y")
        ]
        self.y_pred_name = "y_pred"
        self.show_num = 20
        self.status = 0

        # variables
        self.df, self.df_z = None, None
        self.idx_names, self.split_loc = None, None
        self.model, self.config = None, None
        self.y_plot_panel, self.decision_panel, self.dep_panel = None, None, None
        self.shap_values, self.explainer, self.decision_values, self.impo_list = None, None, None, None
        self.source, self.source_df = None, None
        self.his_table_source, self.record_table_source = None, None
        self.drop_cols = None
        self.data_table = None
        self.group_idx_name, self.data_index = None, None
        self.y_pred = None
        self.history = None
        self.correct_order = None
        self.pred_text = None
        self.slider_dict_uncon, self.slider_dict_con = None, None
        self.text_dict_uncon, self.text_dict_con = None, None
        self.p1_mapper, self.c1_mapper, self.mapper = None, None, None
        self.color_df, self.bins = None, None
        self.history_dict, self.record_table = None, None
        self.table_panel = list()
        self.uncon_panel = list()
        self.con_panel = list()
        self.pred_panel = list()
        self.his_panel = list()

        # execute
        self.__call__()

    def __call__(self, *args, **kwargs):
        self.get_basic_files()
        self.get_table_module()
        self.get_uncontrollable_module()
        self.get_controllable_module()
        self.get_prediction_module()
        self.get_plot_module()
        self.get_history_module()
        self.get_layout()

    def get_basic_files(self):
        self.config = load_yaml(self.config_path)
        self.df = pd.read_csv(self.config['train_data_path'])
        self.correct_order = list(self.df.columns)
        self.model = load_pickle(self.config['model_path'])
        self.group_idx_name = "_".join(self.config['id'])
        self.shap_values = pd.read_csv(self.config['shap_path'])
        self.explainer = load_pickle(self.config['explainer_path'])
        self.decision_values = pd.read_csv(self.config['decision_path'])
        self.impo_list = list(self.decision_values['parameters'])
        self.impo_list.remove('Base')
        self.history = pd.read_csv(self.config['history_path'])
        # self.df = self.astype(self.df)
        self.create_new_id()
        self.get_pred_y()
        self.split_loc = len(self.df) - 0.5
        self.source, self.source_df = self.create_source()

    def create_new_id(self):
        tmp = pd.DataFrame()
        tmp['ID'] = self.df[self.config['id']].apply(lambda x: '_'.join(str(e) for e in x), axis=1)
        self.df = self.df.drop(self.config['id'], axis=1)
        self.df = pd.concat([self.df, tmp], axis=1)

    def get_pred_y(self):
        self.df_z = self.data_transform(self.df)
        self.drop_cols = [self.config['y_name'], self.group_idx_name]
        y_pred_z = self.model_predict(self.model, self.df_z.drop(self.drop_cols, axis=1))
        self.y_pred = self.data_inv_transform(pd.DataFrame(data=y_pred_z, columns=[self.config['y_name']]),
                                              col=self.config['y_name']).values.flatten()

    def astype(self, df):
        for col in self.config['id']:
            df[col] = df[col].astype('str')
        return df

    @staticmethod
    def data_transform(data, col: list = None):
        '''one can transform their data in this function
        col: the columns desired to be transformed, must be list
        '''
        data_z = data
        return data_z

    @staticmethod
    def data_inv_transform(data_z, col: list = None):
        '''one can inverse transform their data in this function
        col: the columns desired to be inverse transformed, must be list
        '''
        data = data_z
        return data

    @staticmethod
    def model_predict(model, df):
        if df.shape[0] == 1:
            data = df.values.reshape(1, -1)
        else:
            data = df.values
        return model.predict(data).ravel()

    def create_source(self):
        data = pd.DataFrame()
        x_data = self.df.drop(self.config['y_name'], axis=1)
        for col in x_data.columns:
            if col == self.group_idx_name:
                data["ID"] = list(x_data[col].values) + ['new_0']
            else:
                data[col] = list(x_data[col].values) + [x_data[col].values[-1]]
                dep = get_dependence_values(col, self.shap_values)
                data["dependence_" + col] = list(dep) + [dep[-1]]

        data[self.config['y_name']] = list(self.df[self.config['y_name']].values) + \
                                      [self.df[self.config['y_name']].values[-1]]
        data[self.y_pred_name] = list(self.y_pred) + [self.y_pred[-1]]
        cols = [x for x in self.decision_values.columns if x not in ["y_pos", "parameters"]]
        data["decision_x"] = [list(self.decision_values[col].values) for col in cols] + \
                             [list(self.decision_values[cols[-1]].values)]
        data["decision_y"] = [list(self.decision_values["y_pos"].values)] * len(data["decision_x"])
        data["index"] = list(x_data.index.values) + [x_data.index.values[-1] + 1]
        # data['attr'] = ['Train'] * len(self.df) + ['Test']
        data['status'] = ['Train'] * len(self.df) + ['Test']
        cmap = plt.get_cmap('RdYlBu')
        y_min, y_max = np.min(data[self.y_pred_name]), np.max(data[self.y_pred_name])
        color_list = [mc.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        self.color_df, self.bins = self.get_color_bins(y_min, y_max, color_list)
        data['color'] = ['#555555'] * len(self.df) + self.get_line_color([self.y_pred[-1]])
        source = ColumnDataSource(data.to_dict(orient='list'))
        return source, data

    @staticmethod
    def get_color_bins(y_min, y_max, color_list):
        df = pd.DataFrame()
        df['a'] = np.linspace(start=y_min, stop=y_max, num=len(color_list))
        df['bucket'], bins = pd.cut(df['a'], bins=len(color_list), retbins=True)
        df['color'] = color_list
        return df, bins

    # def get_line_color(self, y_pred):
        # num = len(y_pred)
        # y_ind_list = list(range(num))
        # color_list = self.get_color_bins(num)
        # _, y_ind_list = (list(t) for t in
        #                  zip(*sorted(zip(y_pred, y_ind_list))))  # sort by y_pred_list and return y_ind_list
        # _, color_list = (list(t) for t in
        #                  zip(*sorted(zip(y_ind_list, color_list))))  # sort by y_ind_list and return color_list
        # return color_list

    def get_line_color(self, y_pred):
        tmp_df = pd.DataFrame()
        tmp_df['y_pred'] = y_pred
        tmp_df['bucket'] = pd.cut(tmp_df['y_pred'], bins=self.bins, retbins=False)
        tmp_df = pd.merge(left=tmp_df, right=self.color_df, on='bucket', how='left')
        mask = (tmp_df['bucket'].isna()) & (tmp_df['y_pred'] <= self.color_df['a'].min())
        tmp_df.loc[mask, 'color'] = tmp_df.loc[mask, 'color'].fillna(self.color_df.loc[0, 'color'])
        mask = (tmp_df['bucket'].isna()) & (tmp_df['y_pred'] >= self.color_df['a'].max())
        tmp_df.loc[mask, 'color'] = tmp_df.loc[mask, 'color'].fillna(self.color_df.loc[len(self.color_df)-1, 'color'])
        # print(tmp_df.tail())
        return list(tmp_df['color'])

    def get_train_test_line_color(self, source_df):
        train_len = len(source_df[source_df['status'] == 'Train'])
        return ['#555555'] * train_len + self.get_line_color(source_df[self.y_pred_name][train_len:])

    def get_uncontrollable_module(self):
        self.slider_dict_uncon = dict()
        self.text_dict_uncon = dict()
        title = Div(text='Uncontrollables', sizing_mode="stretch_width",
                    style={'font-size': '150%', 'color': self.txt_color})
        self.uncon_panel.append(title)
        for col in self.df.columns:
            if col not in [self.config['y_name'], *self.config['id']] and col not in self.config['controllables']:
                self.cols_core(col, self.slider_dict_uncon, self.text_dict_uncon, self.uncon_panel)

    def get_controllable_module(self):
        self.slider_dict_con = dict()
        self.text_dict_con = dict()
        title = Div(text='Controllables', sizing_mode="stretch_width",
                    style={'font-size': '150%', 'color': self.txt_color})
        self.con_panel.append(title)
        for col in self.config['controllables']:
            self.cols_core(col, self.slider_dict_con, self.text_dict_con, self.con_panel)
        con_button = Button(label="Get Advice", button_type="success", width=150, width_policy='fit')
        self.con_panel.append(con_button)

    def cols_core(self, col, slider_dict, text_dict, param_panel):
        slider_dict[col] = Slider(title=col, value=self.source_df[col].values[-1],
                                  start=self.df[col].min(), end=self.df[col].max(),
                                  step=step_dict[col], width=120, width_policy='fit', show_value=False,
                                  margin=(0, 0, 0, 20))
        text_dict[col] = TextInput(value=str(self.source_df[col].values[-1]), width=80, width_policy='fit',
                                   background=self.bck_color)
        slider_dict[col].on_change('value', self.update_text(slider_dict, text_dict, col))
        text_dict[col].on_change('value', self.update_slider(slider_dict, text_dict, col))
        param_panel.append(row(slider_dict[col], text_dict[col]))

    def get_prediction_module(self):
        title = Div(text='Prediction', sizing_mode="stretch_width",
                    style={'font-size': '150%', 'color': self.txt_color})
        self.pred_text = TextInput(value='{:.4f}'.format(self.source_df['decision_x'].values[-1][0]), width_policy='fit', width=150)
        pred_button = Button(label="Get Prediction", button_type="success",  width_policy='fit', width=150)
        pred_button.on_click(self.pred_click)
        self.pred_panel = [title, self.pred_text, pred_button]

    def pred_click(self):
        y_pred, pred_df = self.get_pred_value()
        self.pred_text.value = "{0:.4f}".format(y_pred[0])
        df_shap, decision_values, idx_names = self.get_shap_info(pred_df)
        self.pred_click_plot(pred_df, y_pred, df_shap, decision_values, idx_names)
        self.get_pred_value_in_his(pred_df)

    def get_pred_value(self):
        param_values = list()
        for col in self.df.columns:
            if col in self.drop_cols:
                continue
            if col in self.slider_dict_con.keys():
                param_values.append(self.slider_dict_con[col].value)
            elif col in self.slider_dict_uncon.keys():
                param_values.append(self.slider_dict_uncon[col].value)
            else:
                raise KeyError('col not in df: {}'.format(col))
        pred_df = pd.DataFrame(data=[param_values], columns=[x for x in self.df.columns if x not in self.drop_cols])
        return self.model_predict(self.model, pred_df), pred_df

    def get_pred_value_in_his(self, pred_df):
        pred_list = []
        for i in range(len(self.history)):
            tmp_df = pred_df.copy()
            for col in self.config['controllables']:
                tmp_df[col] = self.history[col][i]
            pred_list.append("{0:.4f}".format(self.model_predict(self.model, tmp_df)[0]))
        self.history_dict['pred.'] = pred_list
        self.his_table_source.data = self.history_dict

    def get_shap_info(self, x_data):
        shap_values = self.explainer.shap_values(x_data)
        idx_names = ['new_' + str(i) for i in range(len(x_data))]
        df_shap = pd.DataFrame(data=shap_values, columns=x_data.columns)
        # importance_list = get_importance_list(df_shap).Features.to_list()
        decision_values = decision_value_core(df_shap, self.explainer.expected_value,
                                              idx_names=idx_names,
                                              importance_list=self.impo_list)
        return df_shap, decision_values, idx_names

    def pred_click_plot(self, x_data, y_pred, df_shap, decision_values, idx_names):
        self.source_df = self.source_df[self.source_df['status'] != 'Test'].reset_index(drop=True)
        tmp_df = pd.DataFrame()
        tmp_df["ID"] = idx_names
        for col in x_data.columns:
            tmp_df[col] = x_data[col].values
            tmp_df["dependence_" + col] = get_dependence_values(col, df_shap)
        tmp_df[self.config['y_name']] = np.array([np.nan]*len(x_data))
        cols = [x for x in decision_values.columns if x not in ["y_pos", "parameters"]]
        tmp_df[self.y_pred_name] = y_pred
        tmp_df["decision_x"] = [list(decision_values[col].values) for col in cols]
        tmp_df["decision_y"] = [self.source_df["decision_y"].values[-1] * len(x_data)]
        tmp_df["index"] = list(x_data.index.values + self.source_df["index"].values[-1] + 1)
        # tmp_df["attr"] = ['Test_{}'.format(x) for x in tmp_df["index"]]
        # tmp_df["attr"] = ['Test1'] * len(x_data)
        tmp_df['status'] = ['Test'] * len(x_data)
        tmp_df2 = pd.concat([self.source_df[[self.y_pred_name, 'status']], tmp_df[[self.y_pred_name, 'status']]])
        tmp_df['color'] = self.get_train_test_line_color(tmp_df2)[-len(x_data):]
        self.source_df = self.source_df.append(tmp_df).reset_index(drop=True)
        self.source.data = self.source_df.to_dict(orient='list')
        # self.new_mapper()
        # https://stackoverflow.com/questions/54428355/bokeh-plot-not-updating
        # https://stackoverflow.com/questions/59041774/bokeh-server-plot-not-updating-as-wanted-also-it-keeps-shifting-and-axis-inform

    # def new_mapper(self):
    #     tmp_df = self.source_df[self.source_df['status'] != 'Train']
    #     test_list = list(tmp_df['attr'])
    #     test_list = [str(x) for x in test_list]
    #     y_pred_list = list(tmp_df[self.y_pred_name])
    #     test_num = len(y_pred_list)
    #     y_ind_list = list(range(test_num))
    #     color_list = self.get_color_bins(test_num)
    #     _, y_ind_list = (list(t) for t in zip(*sorted(zip(y_pred_list, y_ind_list))))  # sort by y_pred_list and return y_ind_list
    #     _, color_list = (list(t) for t in zip(*sorted(zip(y_ind_list, color_list))))  # sort by y_ind_list and return color_list
    #     cat = ['Train'] + test_list
    #     cat_color = ['#555555'] + color_list
    #     self.p1_mapper = factor_cmap('attr', cat_color, cat)
    #     cat_color = ['#555555', 'orange']
    #     self.c1_mapper = factor_cmap('attr', cat_color, cat)

    def get_history_module(self):
        self.history_dict = {k: list(self.history[k]) for k in self.config['controllables']}
        self.history_dict['pred.'] = [''] * len(self.history)
        # self.history_dict['imple.'] = ['V'] * len(self.history)
        self.his_table_source = ColumnDataSource(self.history_dict)
        cols = [TableColumn(field=k, title=k) for k in self.config['controllables']]
        # cols.extend([TableColumn(field='pred.', title='pred.'), TableColumn(field='imple.', title='imple.')])
        cols.extend([TableColumn(field='pred.', title='pred.')])
        his_table = DataTable(source=self.his_table_source, columns=cols,
                              width=200, height=800, editable=True, reorderable=False)
        title = Div(text='History', sizing_mode="stretch_width",
                    style={'font-size': '150%', 'color': self.txt_color})
        self.his_panel.append(title)
        self.his_panel.append(his_table)

    def record_table_callback(self, test_cell, test_count):
        def table_callback(attr, old, new):
            # test_count.value is used to prevent delete_record_row/show_record_row from executing twice
            if int(test_count.value) % 2 == 0:
                # new gives the selected row index
                exp_ind = self.record_table_source.data['exp'][new[0]]  # exp_ind are exp_00, exp_01,...
                if test_cell.value == 'X':
                    self.delete_record_row(exp_ind)
                else:
                    self.show_record_row(exp_ind)
            test_count.value = str(int(test_count.value) + 1)
            self.record_table_source.selected.update(indices=[])  # deselect the table
        return table_callback

    def get_table_module(self):
        # https://stackoverflow.com/questions/54426404/bokeh-datatable-return-row-and-column-on-selection-callback
        self.record_cols = [*list(self.df.columns), 'exp', 'delete', 'his_pred', self.y_pred_name, 'index']
        self.record_table = pd.DataFrame(columns=self.record_cols)
        self.record_table_source = ColumnDataSource(self.record_table[['exp', 'delete']].to_dict(orient='list'))
        cols = [TableColumn(field="exp", title="Exp."),
                TableColumn(field="delete", title="Delete")]
        data_table = DataTable(source=self.record_table_source, columns=cols,
                               width=200, height=800, editable=True, reorderable=False)
        test_cell = TextInput(value='kkk', title="Cell Contents:", width=200)
        test_count = TextInput(value='0', title="Cell Contents:", width=200)
        source_code = """
        var grid = document.getElementsByClassName('grid-canvas')[0].children;
        var row, column = '';

        for (var i = 0,max = grid.length; i < max; i++){
            if (grid[i].outerHTML.includes('active')){
                row = i;
                for (var j = 0, jmax = grid[i].children.length; j < jmax; j++)
                    if(grid[i].children[j].outerHTML.includes('active')) 
                        { column = j }
            }
        }
        test_cell.value = column == 1 ? String(source.data['exp'][row]) : String(source.data['delete'][row]);
        """
        callback = CustomJS(args=dict(source=self.record_table_source, test_cell=test_cell),
                            code=source_code)
        self.record_table_source.selected.js_on_change('indices', callback)
        self.record_table_source.selected.on_change('indices', self.record_table_callback(test_cell, test_count))
        title = Div(text='Record', sizing_mode="stretch_width",
                    style={'font-size': '150%', 'color': self.txt_color})
        record_button = Button(label="Take a snapshot", button_type="success", width_policy='fit', width=150)
        record_button.on_click(self.record)
        self.table_panel.append(title)
        # self.table_panel.append(test_cell)
        self.table_panel.append(record_button)
        self.table_panel.append(data_table)

        # https://stackoverflow.com/questions/55403853/how-to-get-a-list-of-bokeh-widget-events-and-attributes-which-can-be-used-to-tr

    def delete_record_row(self, exp_ind):
        data_ind = self.record_table.loc[self.record_table['exp'] == exp_ind, 'index'].values[0]
        self.record_table = self.record_table[~self.record_table['exp'].isin([exp_ind])]
        self.record_table_source.data = self.record_table[['exp', 'delete']].to_dict(orient='list')
        self.source_df = self.source_df[self.source_df['index'] != data_ind].reset_index(drop=True)
        self.source_df['index'] = list(self.source_df.index)
        self.source.data = self.source_df.to_dict(orient='list')

    def show_record_row(self, exp_ind):
        tmp_df = self.record_table[self.record_table['exp'].isin([exp_ind])]
        tmp_df = tmp_df.drop(['exp', 'delete', self.config['y_name']], axis=1)
        for col in tmp_df.columns:
            if col in [self.group_idx_name, 'index']:
                continue
            elif col == self.y_pred_name:
                self.pred_text.value = "{0:.4f}".format(tmp_df[col].values[0])
            elif col in self.slider_dict_con.keys():
                self.slider_dict_con[col].value = tmp_df[col].values[0]
                self.text_dict_con[col].value = "{0:.4f}".format(tmp_df[col].values[0])
            elif col in self.slider_dict_uncon.keys():
                self.slider_dict_uncon[col].value = tmp_df[col].values[0]
                self.text_dict_uncon[col].value = "{0:.4f}".format(tmp_df[col].values[0])
            elif col == 'his_pred':
                self.history_dict['pred.'] = tmp_df[col].values[0]
                self.his_table_source.data = self.history_dict
            else:
                raise KeyError('col not in df: {}'.format(col))

    def record(self):
        self.record_in_table()

    def record_in_table(self):
        if 'Test' in self.source_df['status'].unique():
            tmp_df = pd.DataFrame()
            for col in self.record_cols:
                if col in self.source_df.columns:
                    tmp_df[col] = [self.source_df[self.source_df['status'] == 'Test'][col].values[0]]
                elif col == 'his_pred':
                    tmp_df[col] = [list(self.history_dict['pred.'])]
            if len(self.record_table['exp'].tail(1)) == 0:
                tmp_df['exp'] = ['exp_00']
            else:
                num = int(self.record_table['exp'].tail(1).values[0][-2:])
                tmp_df['exp'] = ['exp_{0:02d}'.format(num+1)]
            tmp_df['delete'] = ['X']
            self.record_table = self.record_table.append(tmp_df)
            self.record_table_source.data = self.record_table[['exp', 'delete']].to_dict(orient='list')
            self.source_df['status'] = self.source_df['status'].str.replace('Test', 'Record')

    def get_attribute(self, obj):
        obj.title.text_font_size = '11pt'
        obj.title.text_color = self.txt_color
        obj.xgrid.grid_line_alpha = 0.2
        obj.ygrid.grid_line_alpha = 0.2
        obj.background_fill_color = self.bck_color
        obj.background_fill_alpha = 0.5
        obj.border_fill_color = self.bck_color
        obj.xaxis.axis_line_color = self.txt_color
        obj.yaxis.axis_line_color = self.txt_color
        obj.xaxis.major_label_text_color = self.txt_color
        obj.yaxis.major_label_text_color = self.txt_color
        obj.xaxis.major_tick_line_color = self.txt_color
        obj.yaxis.major_tick_line_color = self.txt_color
        obj.xaxis.minor_tick_line_color = self.txt_color
        obj.yaxis.minor_tick_line_color = self.txt_color
        return obj

    # @staticmethod
    # def get_color_bins(n):
    #     cmap = plt.get_cmap('RdYlBu')
    #     color_list = [mc.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    #     color_dict = {ind: color for ind, color in enumerate(color_list)}
    #     # cmap.N = 256 in RdYlBu
    #     if n == cmap.N:
    #         return color_list
    #     elif n > cmap.N:
    #         left = n % cmap.N
    #         tmp_list = list(range(cmap.N))
    #         tmp_list = tmp_list + tmp_list[:left]
    #         tmp_list.sort()
    #         return [color_dict[x] for x in tmp_list]
    #     else:
    #         tmp_list = np.random.uniform(low=0, high=cmap.N-1, size=n)
    #         tmp_list = [int(x) for x in tmp_list]
    #         tmp_list.sort()
    #         return [color_dict[x] for x in tmp_list]

    def get_y_module(self):
        p2 = figure(title="Prediction Result", tools=self.tools, tooltips=self.tooltips, plot_width=500, plot_height=200)
        p2 = self.get_attribute(p2)
        c1 = p2.line('index', self.y_pred_name, legend_label="pred Y", color="#C9C462", name="pred Y",
                     source=self.source, line_width=1.5)
        c1.nonselection_glyph = Line(line_color="#C9C462", line_alpha=0.2)  # when selected, change alpha
        # c1.selection_glyph = Line(line_color="#C9C462", line_width=1.5)

        c2 = p2.circle('index', self.config['y_name'], source=self.source, name="real Y", legend_label="real Y",
                       size=5, color="#FF4040")
        c2.nonselection_glyph = Circle(fill_color="#FF4040", fill_alpha=0.2, line_color=None, size=5)
        # c2.selection_glyph = Circle(fill_color='orange', line_color=None, size=10, line_width=15)

        c2 = p2.circle('index', self.y_pred_name, source=self.source, name="pred Y", size=5, color="#C9C462")
        c2.nonselection_glyph = Circle(fill_color="#C9C462", fill_alpha=0.2, line_color=None, size=5)
        vline = Span(location=self.split_loc, dimension='height', line_color='SeaGreen', line_width=3)
        p2.add_layout(vline)
        p2.legend.location = "top_left"
        p2.legend.click_policy = "hide"
        p2.legend.background_fill_alpha = 0.8
        return p2

    def get_mapper(self):
        self.mapper = linear_cmap(self.y_pred_name, RdYlBu[9],
                                  self.source_df[self.y_pred_name].values.min(),
                                  self.source_df[self.y_pred_name].values.max())

    def get_decision_module(self):
        decision_name = "Decision Plot (avg = {0:.2f})".format(self.explainer.expected_value[0])
        if self.decision_values["y_pos"].max()-self.show_num < 0:
            y_range_min = 0
        else:
            y_range_min = self.decision_values["y_pos"].max()-self.show_num
        p1 = figure(title=decision_name, tools=self.tools, tooltips=self.tooltips, plot_width=400, plot_height=800,
                    x_axis_location='above', y_range=(y_range_min,
                                                      self.decision_values["y_pos"].max()))
        p1 = self.get_attribute(p1)
        # c2 = p1.multi_line(xs='decision_x', ys='decision_y', source=self.source, color=self.p1_mapper, line_width=1.5,
        #                    name='decision plot')
        c2 = p1.multi_line(xs='decision_x', ys='decision_y', source=self.source, line_color='color', line_width=1.5,
                           name='decision plot')
        c2.nonselection_glyph = MultiLine(line_color='#555555', line_alpha=0.2, line_width=1.5)
        c2.selection_glyph = MultiLine(line_color=self.mapper, line_width=1.5)
        # c2.selection_glyph = MultiLine(line_color='orange', line_width=2)
        y_pos = self.decision_values["y_pos"].values - 0.5
        y_pos = y_pos[:-1]
        labels = self.decision_values["parameters"].values[:-1]
        p1.yaxis.ticker = y_pos
        p1.yaxis.major_label_overrides = pd.Series(labels, index=y_pos).to_dict()
        p1 = self.get_attribute(p1)
        return p1

    def get_dep_module(self):
        dep_picture = []
        subtitle = " (dep. plot)"
        show_list = self.decision_values['parameters'][:self.show_num].to_list()
        show_list.remove('Base')
        for col in show_list:
            tmp_p = figure(title=col + subtitle, tools=self.tools, tooltips=self.tooltips,
                           plot_width=500, plot_height=150)
            tmp_p = self.get_attribute(tmp_p)
            c1 = tmp_p.circle(col, "dependence_" + col, source=self.source, name="dep_" + col, size=3.5,
                              # color="ForestGreen")
                              # color=self.c1_mapper)
                              color='color')
            c1.nonselection_glyph = Circle(fill_color='gray', fill_alpha=0.2, line_color=None, size=5)
            c1.selection_glyph = Circle(fill_color='orange', line_color=None, size=10, line_width=15)
            dep_picture.append(tmp_p)
        return dep_picture

    def get_plot_module(self):
        self.y_plot_panel = self.get_y_module()
        #self.get_mapper()
        #self.new_mapper()
        self.decision_panel = self.get_decision_module()
        self.dep_panel = self.get_dep_module()

    def get_each_module(self):
        # table_panel = column(*self.table_panel)
        # con_panel = column(*self.con_panel, *self.pred_panel, width=260)
        # uncon_panel = column(*self.uncon_panel)
        # plot_panel = column(self.y_plot_panel, row(self.decision_panel, column(*self.dep_panel)))
        # return table_panel, con_panel, uncon_panel, plot_panel
        con_panel = column(*self.con_panel, *self.pred_panel, *self.table_panel, width=260)
        uncon_panel = column(*self.uncon_panel)
        plot_panel_1 = column(self.decision_panel)
        plot_panel_2 = column(gridplot(children=[[x] for x in [self.y_plot_panel, *self.dep_panel]],
                                       toolbar_location='right'))
        his_panel = column(*self.his_panel)
        return con_panel, uncon_panel, plot_panel_1, plot_panel_2, his_panel

    def get_layout(self):
        # table_panel, con_panel, uncon_panel, plot_panel = self.get_each_module()
        con_panel, uncon_panel, plot_panel_1, plot_panel_2, his_panel = self.get_each_module()
        desc = Div(text=open(self.config['description_path']).read(), sizing_mode="stretch_width")
        l1 = layout([[uncon_panel, Spacer(width=20),
                      con_panel, Spacer(width=20),
                      plot_panel_1, Spacer(width=20),
                      plot_panel_2, Spacer(width=20),
                      his_panel]])
        # l1 = layout([[table_panel, Spacer(width=20),
        #               uncon_panel, Spacer(width=20),
        #               con_panel, Spacer(width=20),
        #               plot_panel, Spacer(width=20)]])
        # l2 = layout([[fig3]], sizing_mode='fixed')

        tab1 = Panel(child=l1, title="This is Tab 1")
        # tab2 = Panel(child=l2, title="This is Tab 2")
        # tabs = Tabs(tabs=[tab1, tab2])
        tabs = Tabs(tabs=[tab1])
        l = layout([
            [desc],
            [tabs]
        ], sizing_mode="scale_both")
        curdoc().add_root(l)
        curdoc().title = "What-if tool"

    @staticmethod
    def update_text(slider_d, text_d, column_name):
        def update_col_text(attrname, old, new):
            text_d[column_name].value = str(slider_d[column_name].value)
        return update_col_text

    @staticmethod
    def update_slider(slider_d, text_d, column_name):
        def update_col_slider(attrname, old, new):
            slider_d[column_name].value = float(text_d[column_name].value)
        return update_col_slider

# python -m bokeh serve What_if_tool --dev What_if_tool