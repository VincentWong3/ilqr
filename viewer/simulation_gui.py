import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

def simulation_view():
    # 示例数据
    x = [1, 2, 3, 4, 5]
    y = [7, 5, 3, 2, 6]
    source = ColumnDataSource(data={'x': x, 'y': y})

    # 创建图表
    p = figure(title="Simulation Page", width=800, height=400)
    p.line('x', 'y', source=source, line_width=2, color="red")

    # 创建页面布局
    layout = pn.Column(p, pn.pane.Markdown("Simulation Page Content"))
    return layout
