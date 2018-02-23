import ipywidgets as widgets

percentage_widget = widgets.FloatSlider(
    value=50,
    min=50,
    max=100.0,
    step=0.1,
    description='Correct [%]:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

trading_cost_widget = widgets.IntSlider(
    value=1,
    min=0,
    max=10,
    step=1,
    description='Cost [bp]:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
)

interest_widget = widgets.IntSlider(
    value=2,
    min=0,
    max=10,
    step=1,
    description='Interest [%]:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
)