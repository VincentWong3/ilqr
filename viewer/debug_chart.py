import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Arrow, NormalHead, Select

class DebugChart:
    def __init__(self, title="Debug Chart", width=400, height=800):
        self.title = title
        self.width = width
        self.height = height

        # 创建数据源
        self.source = ColumnDataSource(data=dict(x=[], y=[]))

        # 创建图表
        self.plot = figure(title=self.title, width=self.width, height=self.height,
                           x_range=(-20, 150), y_range=(-10, 10))

        # 绘制x轴和y轴
        self.plot.add_layout(Arrow(end=NormalHead(size=10), x_start=0, y_start=0, x_end=0, y_end=9, line_width=2))
        self.plot.add_layout(Arrow(end=NormalHead(size=10), x_start=0, y_start=0, x_end=149, y_end=0, line_width=2))

        # 添加标签
        self.plot.text(x=[0, 149], y=[9, 0], text=["X (North)", "Y (West)"], text_font_size="10pt", text_align="center", text_baseline="middle")

        self.plot.circle('x', 'y', source=self.source)

    def update_data(self, t):
        x = np.linspace(-20, 150, 100)
        y = np.sin(x + t)
        self.source.data = dict(x=x, y=y)

    def get_plot(self):
        return self.plot

def create_debug_chart_view():
    x = np.linspace(-20, 150, 100)
    y = np.sin(x)

    debug_chart = DebugChart(title="Debug Chart with Custom Axes")
    debug_chart.update_data(0)

    # 创建下拉菜单
    select_model = Select(title="Select Vehicle Dynamics Model", value="lat_kinematic_model", options=[
        "lat_kinematic_model", "lon_model", "lat_dynamic_model", "full_kinematic_model", "full_dynamic_model"])

    def update_model(attr, old, new):
        print(f"Selected model: {new}")

    select_model.on_change('value', update_model)

    layout = pn.Column(select_model, debug_chart.get_plot())
    return layout, debug_chart
