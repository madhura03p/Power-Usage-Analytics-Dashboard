#################################
### IMPORT RELEVANT LIBRARIES ###
#################################
# PYTHON VERSION 3.11
import dash  # v2.18.0
import dash_bootstrap_components as dbc  # v1.6.0
import pandas as pd  # v2.2.1
import plotly.express as px  # v5.24.0
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from statsmodels.tsa.seasonal import seasonal_decompose  # v0.14.4


###########################
### DATABASE CONNECTION ###
###########################

# This is for making executable file creation
import os
import sys
def resource_path(relative_path):
    try:
        # When running as an executable
        base_path = sys._MEIPASS
    except AttributeError:
        # When running from the source code
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Use the helper function to load the CSV file
csv_file_path = resource_path(' ')
df = pd.read_csv(csv_file_path)

# This is for testing
# data = " "
# df = pd.read_csv(data)

#############################
### DATABASE MANIPULATION ###
#############################
df = df.drop(
    columns=['StnNo', 'VR', 'VY', 'VB', 'CR', 'CY', 'CB', 'FQ', 'KWH', 'ExportKWH', 'ImportKVAH', 'ExportKVAH', 'KW',
             'DiffExportKVAH', 'DiffImportKVAH'])
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H:%M')
df['Day'] = df['DateTime'].dt.day_name()
df['Date'] = df['DateTime'].dt.date
df['Time'] = df['DateTime'].dt.time
df['KWH'] = df['DiffIKWH'] - df['DiffExportKWH']
df['KWH'] = df['KWH'].clip(lower=0)


def assign_shift(time):
    if pd.to_datetime('06:30', format='%H:%M').time() <= time <= pd.to_datetime('15:00',
                                                                                format='%H:%M').time():
        return 1
    elif pd.to_datetime('15:00', format='%H:%M').time() < time <= pd.to_datetime('23:30',
                                                                                 format='%H:%M').time():
        return 2
    else:
        return 3


df['Shift'] = df['DateTime'].dt.time.apply(assign_shift)

##############
### NAVBAR ###
##############
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Back to Top", href="/")),  # Link to go up
        # dbc.NavItem(dbc.NavLink("Forecast", href="/forecast")),  # Link to forecast page
    ],
    brand="Power Monitoring Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
    fixed="top"
)

#####################
### CARD : FILTER ###
#####################

# List for dropdown
substations = [{'label': sub, 'value': sub} for sub in df['Area'].unique()]
substations.insert(0, {'label': 'All Substations', 'value': 'All'})

filter_card = dbc.Card(
    [
        # dbc.CardHeader("Filters"),
        dbc.CardBody(
            [
                # Substation Dropdown
                dbc.Row(
                    [
                        dbc.Col(
                            html.Label("Select Substation(s):", style={'font-weight': 'normal'}),
                            width=4
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='substation-dropdown',
                                options=substations,
                                value='All',
                                multi=True,
                                placeholder="Select Substations",
                            ),
                            width=8
                        ),
                    ],
                    className="mb-3"
                ),
                # Date Selection
                dbc.Row(
                    [
                        dbc.Col(
                            html.Label("Select Date Range:", style={'font-weight': 'normal'}),
                            width=4
                        ),
                        dbc.Col(
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                start_date=df['Date'].min(),
                                end_date=df['Date'].max(),
                                display_format='DD-MM-YYYY',
                            ),
                            width=8
                        ),
                    ],
                    className="mb-3"
                ),
                # Shift Radio Button
                dbc.Row(
                    [
                        dbc.Col(
                            html.Label("Select Shift:", style={'font-weight': 'normal'}),
                            width=4
                        ),
                        dbc.Col(
                            dcc.RadioItems(
                                id='shift-radio',
                                options=[
                                    {'label': 'Shift 1', 'value': 1},
                                    {'label': 'Shift 2', 'value': 2},
                                    {'label': 'Shift 3', 'value': 3},
                                    {'label': 'All Shifts', 'value': 'All'}
                                ],
                                value='All',
                                labelStyle={'display': 'inline-block'},
                            ),
                            width=8
                        ),
                    ],
                    className="mb-3"
                ),
            ]
        ),
    ],
    style={"margin": "20px", "margin-top": "70px"},
)


###########################
### GRAPH 1: LINE GRAPH ###
###########################

# Function to create graph
def create_line_graph(filtered_df):
    total_df = filtered_df.groupby('DateTime', as_index=False)['KWH'].sum()
    daily_df = filtered_df.groupby('Date', as_index=False)['KWH'].sum()

    # Plotting the graph
    actual_fig = px.line(total_df, x='DateTime', y='KWH', labels={'KWH': 'Power in KWH', 'DateTime': 'Time'})
    actual_fig.add_scatter(x=daily_df['Date'], y=daily_df['KWH'], mode='lines+markers', name='Daily Total KWH',
                           line=dict(dash='dash', color='red'))
    return actual_fig


# For layout
pw_usage = dbc.Row(
    dbc.Col(
        html.Div([
            html.H3("Total Power Consumption Over Time"),
            dcc.Graph(id='graph-output')
        ]),
        width=12
    )
)


###################################
### GRAPH 2: STACKED AREA GRAPH ###
###################################
# Function to create graph
def create_stacked_area_graph(filtered_df, num_days):
    if num_days > 7:
        filtered_df['Date'] = filtered_df['Date'].astype(str)
        daily_df = filtered_df.groupby(['Date', 'Area', 'PF'], as_index=False).agg({'KWH': 'sum', 'PF': 'mean'})
        daily_df['DateTime'] = pd.to_datetime(daily_df['Date'])
    else:
        daily_df = filtered_df.copy()

    daily_df['Total_KWH'] = daily_df.groupby('DateTime')['KWH'].transform('sum')
    daily_df['KWH_normalized'] = (daily_df['KWH'] / daily_df['Total_KWH']) * 100

    stacked_area_fig = px.bar(daily_df, x='DateTime', y='KWH_normalized', color='Area',
                              labels={'KWH_normalized': 'Percentage Use', 'DateTime': 'Time'},
                              height=500, title='Stacked Column Graph of KWH Usage by Area')
    return stacked_area_fig


# For layout
stacked_area = dbc.Row(
    dbc.Col(
        html.Div([
            html.H3("Percentage Power Consumed by Substation"),
            dcc.Graph(id='stacked-area-output')
        ]),
        width=12
    )
)


#############################################
### GRAPH 3 AND 4: SHIFT WISE INFORMATION ###
#############################################
# Function to create pie chart
def create_shift_pie_chart(filtered_df):
    shift_df = filtered_df.groupby('Shift', as_index=False)['KWH'].sum()
    shift_labels = {1: 'Shift 1', 2: 'Shift 2', 3: 'Shift 3'}
    shift_df['Shift'] = shift_df['Shift'].map(shift_labels)

    shift_pie_chart_fig = px.pie(shift_df, values='KWH', names='Shift',
                                 hover_data=['KWH'],
                                 labels={'KWH': 'Total KWH'})
    shift_pie_chart_fig.update_traces(textinfo='percent', hovertemplate='%{label}: %{value:.2f} KWH<br>%{percent}')
    return shift_pie_chart_fig


# Function to create clustered bar graph
def create_shift_bar_chart(filtered_df):
    shift_stats = filtered_df.groupby('Shift').agg(
        Avg=('KWH', 'mean'),
        Min=('KWH', 'min'),
        Max=('KWH', 'max')
    ).reset_index()

    shift_labels = {1: 'Shift 1', 2: 'Shift 2', 3: 'Shift 3'}
    shift_stats['Shift'] = shift_stats['Shift'].map(shift_labels)

    overall_avg = filtered_df['KWH'].mean()

    # Make the graph
    shift_bar_fig = px.bar(
        shift_stats,
        x='Shift',
        y=['Avg', 'Min', 'Max'],
        barmode='group',
        labels={'value': 'KWH', 'Shift': 'Shift'},
        title=""
    )
    # Add the average line
    shift_bar_fig.add_shape(
        type='line',
        x0=-0.5,
        y0=overall_avg,
        x1=2.5,
        y1=overall_avg,
        line=dict(color='red', width=2, dash='dash'),
        xref='x',
        yref='y'
    )

    shift_bar_fig.add_annotation(
        x=1,
        y=overall_avg,
        text=f"Avg KWH: {overall_avg:.2f}",
        showarrow=False,
        yshift=10,
        font=dict(color='red')
    )

    shift_bar_fig.update_layout(
        legend_title_text='',
        xaxis_title='Shift',
        yaxis_title='KWH'
    )
    return shift_bar_fig


# For layout
shift_wise_info = dbc.Row([
    dbc.Col(
        html.Div([
            html.H3("Power Consumption Share by Shift"),
            dcc.Graph(id='shift-pie-chart')
        ]),
        width=6
    ),
    dbc.Col(
        html.Div([
            html.H3("Shift-wise Power Details"),
            dcc.Graph(id='shift-bar-output')
        ]),
        width=6
    )
])


#####################################
### GRAPH 5: POWER FACTOR HEATMAP ###
#####################################
# Function to create heatmap
def create_heatmap(filtered_df, num_days):
    if num_days > 7:
        filtered_df['Date'] = filtered_df['Date'].astype(str)
        daily_df = filtered_df.groupby(['Date', 'Area', 'PF'], as_index=False).agg({'KWH': 'sum', 'PF': 'mean'})
        daily_df['DateTime'] = pd.to_datetime(daily_df['Date'])
        pf_df = daily_df.copy()
    else:
        pf_df = filtered_df.copy()
    pf_df['Hour'] = pf_df['DateTime'].dt.hour
    pf_df = pf_df.groupby(['Date', 'Hour', 'Area'], as_index=False)['PF'].mean()
    pf_df['DateTime'] = pd.to_datetime(pf_df['Date']) + pd.to_timedelta(pf_df['Hour'], unit='h')
    pf_pivot = pf_df.pivot_table(index='Area', columns='DateTime', values='PF', aggfunc='mean')

    color_scale = [[0, 'red'], [0.95, 'orange'], [1, 'green']]
    if not pf_pivot.empty:
        heatmap_fig = px.imshow(pf_pivot, color_continuous_scale=color_scale,
                                labels={'x': 'Time', 'y': 'Area', 'color': 'Power Factor'})
    else:
        heatmap_fig = px.imshow([[0]], color_continuous_scale=color_scale, title="No Data Available")
    return heatmap_fig


# For Layout
pf_heatmap = dbc.Row(
    dbc.Col(
        html.Div([
            html.H3("Power Factor Heatmap"),
            dcc.Graph(id='heatmap-output')
        ]),
        width=12
    )
)


######################################################
### GRAPH 6 AND 7: BAR CHART AND PIE CHART IMP/EXP ###
######################################################
# Function to create bar graph
def create_avg_per_day_bar_graph(filtered_df):
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    filtered_df['Day'] = pd.Categorical(filtered_df['Day'], categories=day_order, ordered=True)
    day_avg_df = filtered_df.groupby('Day', as_index=False).agg({'KWH': 'mean', 'PF': 'mean'})

    bar_fig = px.bar(
        day_avg_df,
        x='Day',
        y='KWH',
        labels={'KWH': 'Average KWH per hour', 'Day': 'Day of Week'},
        color_continuous_scale='Viridis'
    )
    bar_fig.update_traces(marker_line_width=2, marker_line_color='black', width=0.2)
    return bar_fig


# Function to create pie chart
def create_import_export_pie_chart(filtered_df):
    import_export_df = filtered_df[['DiffIKWH', 'DiffExportKWH']].sum().reset_index()
    import_export_df.columns = ['Type', 'KWH']
    import_export_df['Type'] = import_export_df['Type'].replace({
        'DiffIKWH': 'Grid KWH', 'DiffExportKWH': 'Solar KWH'
    })

    import_export_pie_fig = px.pie(import_export_df, values='KWH', names='Type',
                                   hover_data=['KWH'],
                                   labels={'KWH': 'KWH'},
                                   color_discrete_sequence=px.colors.qualitative.Set1)
    import_export_pie_fig.update_traces(textinfo='percent')
    return import_export_pie_fig


# For Layout
info_charts = dbc.Row([
    dbc.Col(
        html.Div([
            html.H3("Average Power Consumed Per Hour of the Week"),
            dcc.Graph(id='daywise-bar-output')
        ]),
        width=8
    ),
    dbc.Col(
        html.Div([
            html.H3("Grid VS Solar KWH"),
            dcc.Graph(id='import-export-pie-chart')
        ]),
        width=4
    )
])


#######################################
### GRAPH 8: SEASONALITY AND TRENDS ###
#######################################
# Function to create the graph
def create_trend_seasonality_graph(filtered_df, num_days):
    if num_days < 28:
        trend_seasonality_fig = go.Figure()
        trend_seasonality_fig.add_annotation(
            text="Minimum 28 days need to be selected to get seasonality breakdown.",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="red"),
            x=0.5, y=0.5,  # Center the text
            align="center"
        )
        trend_seasonality_fig.update_layout(
            title="Seasonality Breakdown",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
    else:
        trend_seasonality_df = filtered_df.groupby('DateTime', as_index=False)['KWH'].sum()
        trend_seasonality_df.set_index('DateTime', inplace=True)
        trend_seasonality_df = trend_seasonality_df.resample('D').sum().dropna()

        # Decompose into trend and seasonal components
        decomposition = seasonal_decompose(trend_seasonality_df['KWH'], model='additive', period=7)
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()

        trend_seasonality_fig = px.line(trend_seasonality_df.reset_index(), x='DateTime', y='KWH',
                                        labels={'KWH': 'KWH', 'DateTime': 'Time'})
        trend_seasonality_fig.add_scatter(x=trend.index, y=trend, mode='lines', name='Trend',
                                          line=dict(color='green'))
        trend_seasonality_fig.add_scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonality',
                                          line=dict(dash='dash', color='red'))

    return trend_seasonality_fig


# For Layout
seasonality_and_trends = dbc.Row(
    dbc.Col(
        html.Div([
            html.H3("Trends and Seasonality"),
            dcc.Graph(id='seasonality-chart'),
            # INFORMATION ON WHAT TRENDS AND SEASONALITY IS
            html.Div([
                html.P("The trend line represents the long-term movement or direction of the data over time, "
                       "indicating consistent increases or decreases in values, which can be upward, downward, "
                       "or flat. An upward trend indicates a general increase in values over 7 day period, "
                       "a downward trend indicates a general decrease in values over 7 day period, and a flat trend "
                       "indicates no significant change, reflecting stability in the data. "),
                html.P(
                    "Seasonality refers to periodic fluctuations in time series data that occur at regular intervals, "
                    "often driven by external factors such as weather, holidays, or economic cycles, resulting in "
                    "consistent patterns or trends within specific time frames, like daily, monthly, or quarterly. In "
                    "this data, seasonality is observed on a weekly basis, calculated over a period of 28 days, "
                    "reflecting recurring patterns or fluctuations that repeat every week.")],
                style={'marginTop': '20px', 'fontSize': '14px', 'color': 'black'})
        ]),
        width=12
    )
)

###############################
### INITIALIZE THE DASH APP ###
###############################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN], suppress_callback_exceptions=True)

###########################
### LAYOUT OF THE PAGE ###
##########################
app.layout = html.Div([

    navbar,

    filter_card,

    pw_usage,

    stacked_area,

    shift_wise_info,

    pf_heatmap,

    info_charts,

    seasonality_and_trends

])


####################################
### FUNCTION TO RETURN MAIN PAGE ###
####################################

@app.callback(
    [Output('graph-output', 'figure'),
     Output('stacked-area-output', 'figure'),
     Output('heatmap-output', 'figure'),
     Output('shift-pie-chart', 'figure'),
     Output('daywise-bar-output', 'figure'),
     Output('import-export-pie-chart', 'figure'),
     Output('shift-bar-output', 'figure'),
     Output('seasonality-chart', 'figure')],
    [Input('substation-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('shift-radio', 'value')]
)
#################################
### FUNCTION TO RETURN CHARTS ###
#################################
def update_graphs(selected_substations, start_date, end_date, selected_shift):
    filtered_df = filter_data(df, selected_substations, start_date, end_date, selected_shift)
    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

    line_graph = create_line_graph(filtered_df)
    stacked_area_graph = create_stacked_area_graph(filtered_df, num_days)
    heatmap = create_heatmap(filtered_df, num_days)
    shift_pie_chart = create_shift_pie_chart(filtered_df)
    daywise_bar_graph = create_avg_per_day_bar_graph(filtered_df)
    import_export_pie_chart = create_import_export_pie_chart(filtered_df)
    shift_bar_chart = create_shift_bar_chart(filtered_df)
    trend_seasonality_graph = create_trend_seasonality_graph(filtered_df, num_days)

    return (line_graph, stacked_area_graph, heatmap, shift_pie_chart,
            daywise_bar_graph, import_export_pie_chart, shift_bar_chart,
            trend_seasonality_graph)


##################################
### FUNCTION TO FILTER DATASET ###
##################################

def filter_data(og_df, selected_substations, start_date, end_date, selected_shift):
    # Apply date filters
    og_df['Date'] = pd.to_datetime(og_df['Date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = og_df[(og_df['Date'] >= start_date) & (og_df['Date'] <= end_date)]

    # Apply substation filter
    if selected_substations != 'All':
        filtered_df = filtered_df[filtered_df['Area'].isin(selected_substations)]

    # Apply shift filter
    if selected_shift != 'All':
        filtered_df = filtered_df[filtered_df['Shift'] == selected_shift]

    # Create DateTime column
    filtered_df['DateTime'] = pd.to_datetime(
        filtered_df['Date'].astype(str) + ' ' + filtered_df['Time'].astype(str))

    return filtered_df


###################
### RUN THE APP ###
###################
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
