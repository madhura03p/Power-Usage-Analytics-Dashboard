#################################
### IMPORT RELEVANT LIBRARIES ###
#################################
# PYTHON VERSION 3.11
import dash  # v2.18.0
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd  # v2.2.1
import plotly.express as px  # v5.24.0
import plotly.graph_objects as go
import base64
import io
from statsmodels.tsa.seasonal import seasonal_decompose  # v0.14.4
from statsmodels.tsa.statespace.sarimax import SARIMAX

###############################
### INITIALIZE THE DASH APP ###
###############################
app = dash.Dash(__name__)

# Initialize an empty DataFrame
df = pd.DataFrame()


#############################################
### Function to parse content of csv file ###
#############################################

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))


###########################
### LAYOUT OF THE PAGE ###
##########################
app.layout = html.Div(
    children=[
        html.H1(
            "Power Consumption Analysis Dashboard",
            style={'textAlign': 'center', 'font-weight': 'bold', 'margin-bottom': '20px'}
        ),
        # Container for Inputs
        html.Div(
            children=[
                #### UPLOAD CSV FILE ####
                html.Label("Upload CSV File:", style={'font-weight': 'bold'}),
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('Upload CSV File'),
                    style={'width': '50%', 'margin': '10px', 'display': 'block', 'margin-left': 'auto',
                           'margin-right': 'auto'}
                ),
                ### DATE RANGE SELECTION ###
                html.Label("Select date:", style={'font-weight': 'bold'}),
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=None,
                    end_date=None,
                    display_format='DD-MM-YYYY',
                    style={'margin': '10px', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}
                ),
                ### STATION SELECTION DROPDOWN ###
                html.Label("Select Station(s):", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='station-dropdown',
                    options=[],
                    value=['All'],
                    multi=True,
                    placeholder="Select Stations",
                    style={'width': '70%', 'margin': '10px', 'display': 'block', 'margin-left': 'auto',
                           'margin-right': 'auto'}
                ),
                #### SHIFT SELECTION DROPDOWN ###
                html.Label("Select Shifts:", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='shift-dropdown',
                    options=[],
                    value=['All'],
                    multi=True,
                    placeholder="Select Shifts",
                    style={'width': '70%', 'margin': '10px', 'display': 'block', 'margin-left': 'auto',
                           'margin-right': 'auto'}
                )
            ],
            style={
                'display': 'block',
                'textAlign': 'center',
                'padding': '20px'
            }
        ),
        ### Graphs ####
        dcc.Graph(id='kwh-graph'),
        dcc.Graph(id='stacked-area-graph-output'),
        dcc.Graph(id='shift-pie-chart'),
        dcc.Graph(id='clustered-bar-graph'),
        dcc.Graph(id='avg-per-day-bar-graph'),
        dcc.Graph(id='seasonality-trend-graph'),
        ## Info about trends and seasonality ##
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
            style={'marginTop': '20px', 'fontSize': '18px', 'color': 'black'}),
        dcc.Graph(id='forecast-graph'),
        html.Div(id='forecast-table'),
        html.Div([
            html.P(
                "A lower bound is the lowest predicted power consumed that day and the  upper bound is the highest "
                "possible power consumed that day.")],
            style={'marginTop': '20px', 'fontSize': '18px', 'color': 'black'}),

    ],
    style={
        'textAlign': 'center',
        'padding': '20px'
    }
)


######################################
### CALLBACK TO POPULATE DATAFRAME ###
######################################
@app.callback(
    [Output('station-dropdown', 'options'),
     Output('shift-dropdown', 'options'),
     Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date')],
    [Input('upload-data', 'contents')]
)
######################################
### FUNCTION TO POPULATE DATAFRAME ###
######################################
def update_data(contents):
    global df
    if contents is not None:
        try:
            df = parse_contents(contents)

            #########################
            ### DATA MANIPULATION ###
            #########################
            # Drop null values
            df.replace('NaN', pd.NA, inplace=True)
            df = df.dropna(axis=0, how='all')
            df = df.dropna(axis=1, how='all')

            # Keep Columns of table where 'Date' and 'Shift' are present. This is done so to keep only Shift wise
            # information
            mask = df.apply(
                lambda col: col.astype(str).str.contains('Date').any() or col.astype(str).str.contains('Shift').any(),
                axis=0)
            df = df.loc[:, mask]
            df.reset_index(drop=True, inplace=True)

            # As the names of the stations were spread across 3 columns in the csv file it became '[station name],
            # null, null' so the next two columns needed to be filled in by the station names
            for i in range(1, len(df.columns) - 1, 3):
                value_to_copy = df.iloc[0, i]
                if i + 1 < len(df.columns):
                    df.iloc[0, i + 1] = value_to_copy
                if i + 2 < len(df.columns):
                    df.iloc[0, i + 2] = value_to_copy
            columns_to_keep = df.columns[(df.iloc[0].notna()) | (df.columns[0] == df.columns)]
            df = df.loc[:, columns_to_keep]
            df.reset_index(drop=True, inplace=True)
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

            # Rename each column  as [Station name] : [Shift]
            for i in range(1, len(df.columns)):
                row_0_value = df.iloc[0, i]
                row_1_value = df.iloc[1, i]
                new_name = f"{row_0_value} : {row_1_value}"
                df.rename(columns={df.columns[i]: new_name}, inplace=True)

            # Change date format
            valid_date_rows = df[pd.to_datetime(df.iloc[:, 0], format='%d-%b-%y', errors='coerce').notna()]
            df = pd.concat([valid_date_rows]).drop_duplicates().reset_index(drop=True)
            df = df.dropna(subset=df.columns[1:])

            # Unpivot the dataframe
            df = df.melt(id_vars=['Date'], var_name='Station_Shift', value_name='KWH')
            df[['Station Name', 'Shift']] = df['Station_Shift'].str.split(' : ', expand=True)
            df.drop(columns=['Station_Shift'], inplace=True)

            # Names of new columns
            df = df[['Date', 'Station Name', 'Shift', 'KWH']]
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y').dt.strftime('%Y-%m-%d')
            df['KWH'] = df['KWH'].astype(float)
            # Drop 'Plant Total'
            df = df[df['Station Name'] != 'Plant Total']

            # Standardize shift name
            def replace_shift(shift):
                shift = shift.lower()
                if 'c' in shift:
                    return 3
                elif 'b' in shift:
                    return 2
                elif 'a' in shift:
                    return 1
                else:
                    return shift

            df['Shift'] = df['Shift'].apply(replace_shift)

            # Get unique station names and shifts from the DataFrame
            # this is list for dropdowns with all label added while returning
            stations = df['Station Name'].unique()
            shifts = df['Shift'].unique()

            return ([{'label': 'All', 'value': 'All'}] + [{'label': station, 'value': station} for station in
                                                          stations],
                    [{'label': 'All', 'value': 'All'}] + [{'label': f'Shift {shift}', 'value': shift} for shift in
                                                          shifts],
                    df['Date'].min(),
                    df['Date'].max())
        except Exception as e:
            print(f"Error processing file: {e}")  # Log the error for debugging
            return [], [], None, None  # Return empty options if there's an error

    return [], [], None, None


###############################################
### CALLOUTS FOR GRAPHS ACCORDING TO INPUTS ###
###############################################
@app.callback(
    [Output('kwh-graph', 'figure'),
     Output('stacked-area-graph-output', 'figure'),
     Output('shift-pie-chart', 'figure'),
     Output('clustered-bar-graph', 'figure'),
     Output('avg-per-day-bar-graph', 'figure'),
     Output('seasonality-trend-graph', 'figure'),
     Output('forecast-graph', 'figure'),
     Output('forecast-table', 'children')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('station-dropdown', 'value'),
     Input('shift-dropdown', 'value')]
)
###################################
### FUNCTION TO GENERATE GRAPHS ###
###################################
def update_graph(start_date, end_date, selected_stations, selected_shifts):
    global df
    if df.empty or 'Date' not in df.columns:
        return {}, {}, {}, {}, {}, {}, {}, {}

    ### Initialize the forecast figure and table ###
    forecast_fig = go.Figure()
    forecast_fig.add_annotation(
        text="Minimum 14 days are needed to get prediction.",
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="red"),
        x=0.5, y=0.5,  # Center the text
        align="center"
    )
    forecast_fig.update_layout(
        title="Power forecast for next 7 days",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    forecast_table = "No forecast data available."

    ########################################
    ### FILTER DATAFRAME BASED ON INPUTS ###
    ########################################
    filtered_df = df.copy()
    if selected_shifts != ['All']:
        filtered_df = filtered_df[filtered_df['Shift'].isin(selected_shifts)]

    if selected_stations and selected_stations != ['All']:
        filtered_df = filtered_df[filtered_df['Station Name'].isin(selected_stations)]

    # Filter the DataFrame based on date
    date_filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

    ###########################
    ### GRAPH 1: LINE GRAPH ###
    ###########################
    ### Filter Dataframe ###
    daily_df = date_filtered_df.copy()
    daily_df = daily_df.groupby(['Date', 'Station Name'], as_index=False)['KWH'].sum()
    daily_df['Total_KWH'] = daily_df.groupby('Date')['KWH'].transform('sum')

    ### Create Line Graph ###
    fig = px.line(daily_df, x='Date', y='KWH', color='Station Name',
                  title='Power Consumption Over Time', labels={'KWH': 'KWH', 'Date': 'Date'})
    fig.add_scatter(x=daily_df['Date'].unique(), y=daily_df.groupby('Date')['Total_KWH'].first(),
                    mode='lines', line=dict(dash='dash', color='black'), name='Total Power')

    #######################################
    ### GRAPH 2: NORMALIZED STACK GRAPH ###
    #######################################
    ### Filter dataframe + calculations ###
    normalized_daily_df = date_filtered_df.copy()
    normalized_daily_df = normalized_daily_df.groupby(['Date', 'Station Name'], as_index=False)['KWH'].sum()
    normalized_daily_df['Total_KWH'] = normalized_daily_df.groupby('Date')['KWH'].transform('sum')
    normalized_daily_df['KWH_normalized'] = (normalized_daily_df['KWH'] / normalized_daily_df['Total_KWH']) * 100

    ### Create Stacked Area Chart ###
    stacked_area_fig = px.area(normalized_daily_df, x='Date', y='KWH_normalized',
                               color='Station Name',
                               labels={'KWH_normalized': 'Percentage Use', 'Date': 'Date'},
                               title='Stacked Area Graph of Power Usage by Station',
                               height=500)
    stacked_area_fig.update_layout(legend_traceorder="reversed")
    stacked_area_fig.update_traces(hovertemplate='<b>%{data.name}</b><br>' + 'Percentage: %{y:.1f}%<br>' +
                                                 'Date: %{x}<br>')

    #####################################
    ### GRAPH 3: SHIFT-WISE PIE CHART ###
    #####################################
    ### Filter dataframe + calculations ###
    shift_df = date_filtered_df.copy()
    shift_df = shift_df.groupby('Shift', as_index=False)['KWH'].sum()
    shift_labels = {1: 'Shift 1', 2: 'Shift 2', 3: 'Shift 3'}
    shift_df['Shift'] = shift_df['Shift'].map(shift_labels)

    ### Create Pie Chart ###
    shift_pie_chart_fig = px.pie(shift_df, values='KWH', names='Shift',
                                 hover_data=['KWH'],
                                 labels={'KWH': 'Total KWH'},
                                 title='Power Consumption Share by Shift')
    shift_pie_chart_fig.update_traces(textinfo='percent', hovertemplate='%{label}: %{value:.2f} KWH<br>%{percent}')

    ###############################################################
    ### GRAPH 4: SHIFT-WISE CLUSTERED BAR GRAPH : MIN, AVG, MAX ###
    ###############################################################
    ### Filter dataframe + calculations ###
    shift_stats = date_filtered_df.groupby('Shift').agg(
        Avg=('KWH', 'mean'),
        Min=('KWH', 'min'),
        Max=('KWH', 'max')
    ).reset_index()
    shift_stats['Shift'] = shift_stats['Shift'].map(shift_labels)
    overall_avg = filtered_df['KWH'].mean()

    ### Create Clustered Bar Chart ###
    shift_bar_fig = px.bar(
        shift_stats,
        x='Shift',
        y=['Min', 'Avg', 'Max'],
        barmode='group',
        labels={'value': 'KWH', 'variable': 'KWH Type'},
        title='Shift-wise Power Usage Analysis'
    )
    # Show KWH value above bar
    shift_bar_fig.update_traces(
        texttemplate='%{y:.2f}',
        textposition='outside'
    )

    # Add the average line
    shift_bar_fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(shift_stats['Shift']) - 0.5,
        y0=overall_avg,
        y1=overall_avg,
        line=dict(color="gray", width=2, dash="dash"),
    )
    # Add average line in the index
    shift_bar_fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color="gray", width=2, dash="dash"),
            name=f'Overall Avg: {overall_avg:.2f} KWH'
        )
    )
    # Legend
    shift_bar_fig.update_layout(
        legend_title_text='',
        xaxis_title='Shift',
        yaxis_title='KWH'
    )

    ###########################################
    ### GRAPH 5: DAY-WISE KWH AVG BAR CHART ###
    ###########################################
    ### Filter dataframe + calculations ###
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    per_day_bar_df = date_filtered_df.copy()
    per_day_bar_df['Day'] = pd.to_datetime(per_day_bar_df['Date']).dt.day_name()
    per_day_bar_df['Day'] = pd.Categorical(per_day_bar_df['Day'], categories=day_order, ordered=True)

    day_avg_df = per_day_bar_df.groupby('Day', as_index=False).agg({'KWH': 'mean'})

    ### Create Bar Graph ###
    avg_per_day_bar_fig = px.bar(
        day_avg_df,
        x='Day',
        y='KWH',
        labels={'KWH': 'Avg KWH', 'Day': 'Day of Week'},
        title='Average Power Consumed Per Hour of the Week'
    )
    # Display KWH value above bar
    avg_per_day_bar_fig.update_traces(
        marker_line_width=2,
        marker_line_color='black',
        width=0.2,
        text=day_avg_df['KWH'].round(2),
        textposition='outside',
        texttemplate='%{text}',
    )
    avg_per_day_bar_fig.update_traces(marker_line_width=2, marker_line_color='black', width=0.2)

    # If empty or less than 14 data points
    if filtered_df['Date'].nunique() < 14:
        seasonality_trend_fig = go.Figure()
        seasonality_trend_fig.add_annotation(
            text="Minimum 14 days are needed to get seasonality breakdown.",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="red"),
            x=0.5, y=0.5,  # Center the text
            align="center"
        )
        seasonality_trend_fig.update_layout(
            title="Seasonality and Trends Breakdown",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
    else:
        #######################################
        ### GRAPH 6: SEASONALITY AND TRENDS ###
        #######################################
        ### Filter dataframe + calculation ###
        trend_seasonality_df = filtered_df.copy()
        trend_seasonality_df = trend_seasonality_df.groupby('Date', as_index=False)['KWH'].sum()
        trend_seasonality_df['Date'] = pd.to_datetime(trend_seasonality_df['Date'])
        trend_seasonality_df.set_index('Date', inplace=True)
        trend_seasonality_df = trend_seasonality_df.resample('D').sum().dropna()
        decomposition = seasonal_decompose(trend_seasonality_df['KWH'], model='additive', period=7)
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()

        ### Create Line Graph ###
        seasonality_trend_fig = px.line(trend_seasonality_df.reset_index(), x='Date', y='KWH',
                                        labels={'KWH': 'KWH', 'Date': 'Date'})
        seasonality_trend_fig.add_scatter(x=trend.index, y=trend, mode='lines', name='Trend',
                                          line=dict(color='red'))
        seasonality_trend_fig.add_scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonality',
                                          line=dict(dash='dash', color='green'))
        seasonality_trend_fig.update_layout(title="Seasonality and Trend Breakdown")

        #################################
        ### GRAPH 7: SARIMA FORECAST ###
        #################################
        if not trend_seasonality_df.empty:
            # Fit SARIMA model
            model = SARIMAX(trend_seasonality_df['KWH'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            results = model.fit()

            # Forecast the next 7 days
            forecast = results.get_forecast(steps=7)
            forecast_index = pd.date_range(start=trend_seasonality_df.index[-1] + pd.Timedelta(days=1), periods=7,
                                           freq='D')
            forecast_values = forecast.predicted_mean
            forecast_conf_int = forecast.conf_int()
            forecast_conf_int['lower KWH'] = forecast_conf_int['lower KWH'].apply(lambda x: max(x, 0))

            ### Create Graph ###
            forecast_fig = go.Figure()
            forecast_fig.add_trace(
                go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='Forecast', line=dict(color='blue')))
            forecast_fig.add_trace(
                go.Scatter(x=forecast_index, y=forecast_conf_int['lower KWH'], mode='lines', name='Lower Bound',
                           line=dict(color='red', dash='dash')))
            forecast_fig.add_trace(
                go.Scatter(x=forecast_index, y=forecast_conf_int['upper KWH'], mode='lines', name='Upper Bound',
                           line=dict(color='red', dash='dash')))
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast_index,
                    y=forecast_values,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='blue'),
                    marker=dict(size=6)
                )
            )

            # Add lower bound
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast_index,
                    y=forecast_conf_int['lower KWH'],
                    mode='lines+markers',
                    name='Lower Bound',
                    line=dict(color='red', dash='dash'),
                    marker=dict(size=6)
                )
            )

            # Add Upper Bound
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast_index,
                    y=forecast_conf_int['upper KWH'],
                    mode='lines+markers',
                    name='Upper Bound',
                    line=dict(color='red', dash='dash'),
                    marker=dict(size=6)
                )
            )
            forecast_fig.update_layout(title="Power Forecast for Next 7 Days", xaxis_title="Date", yaxis_title="KWH")

            ######################
            ### FORECAST TABLE ###
            ######################
            forecast_table = html.Div(
                html.Table(
                    [html.Tr(
                        [html.Th("Date"), html.Th("Forecasted KWH"), html.Th("Lower Bound"), html.Th("Upper Bound")])] +
                    [html.Tr([html.Td(forecast_index[i].strftime('%Y-%m-%d')),
                              html.Td(f"{forecast_values[i]:.2f}"),
                              html.Td(f"{forecast_conf_int['lower KWH'][i]:.2f}"),
                              html.Td(f"{forecast_conf_int['upper KWH'][i]:.2f}")])
                     for i in range(len(forecast_index))],
                    style={'margin': '0 auto', 'text-align': 'center'}
                ),
                style={'display': 'flex', 'justify-content': 'center'}
            )
        else:
            forecast_table = "No forecast data available."
            forecast_fig = go.Figure()
            forecast_fig.add_annotation(
                text="No forecast data available.",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color="red"),
                x=0.5, y=0.5,  # Center the text
                align="center"
            )
            forecast_fig.update_layout(
                title="Prediction for next 7 days",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )

    return (fig, stacked_area_fig, shift_pie_chart_fig, shift_bar_fig, avg_per_day_bar_fig, seasonality_trend_fig,
            forecast_fig, forecast_table)


###################
### RUN THE APP ###
###################
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
