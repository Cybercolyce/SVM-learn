import pyecharts.options as opts
import numpy as np
from pyecharts.charts import Scatter,Line

x_data1 = [1.0,3.0,]
y_data1 = [1.0,3.0]
x_data2 = [4.0]
y_data2 = [3.0]

(
    Scatter(init_opts=opts.InitOpts(width="200px", height="200px"))
    .add_xaxis(xaxis_data=x_data1)
    .add_yaxis(
        series_name="A",
        y_axis=y_data1,
        symbol_size=20,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_xaxis(xaxis_data=x_data2)
    .add_yaxis(
        series_name="B",
        y_axis=y_data2,
        symbol_size=20,
        symbol='diamond',
        label_opts=opts.LabelOpts(is_show=False),

    )
    .set_series_opts()
    .set_global_opts(
        title_opts=opts.TitleOpts(title="一个简单的例子",pos_left="center"),
        xaxis_opts=opts.AxisOpts(
            type_="value", splitline_opts=opts.SplitLineOpts(is_show=True),
            min_='0',
            max_='5',
            interval=0.5
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
            min_='0',
            max_='5',
            interval=0.5
        ),
        legend_opts=opts.LegendOpts(pos_bottom='bottom')
    )
    .render("Example.html")
)

#############################################################
x_data11=np.arange(0,4.1,0.1)
x_data22=np.arange(0,2.1,0.1)
x_data33=np.arange(0,6.1,0.1)
y_data11=[]
y_data22=[]
y_data33=[]
scatter=(
    Scatter(init_opts=opts.InitOpts(width="200px", height="200px"))
    .add_xaxis(xaxis_data=x_data1)
    .add_yaxis(
        series_name="A",
        y_axis=y_data1,
        symbol_size=20,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_xaxis(xaxis_data=x_data2)
    .add_yaxis(
        series_name="B",
        y_axis=y_data2,
        symbol_size=20,
        symbol='diamond',
        label_opts=opts.LabelOpts(is_show=False),

    )
    .set_series_opts()
    .set_global_opts(
        title_opts=opts.TitleOpts(title="一个简单的例子",pos_left="center"),
        xaxis_opts=opts.AxisOpts(
            type_="value", splitline_opts=opts.SplitLineOpts(is_show=True),
            min_='0',
            max_='5',
            interval=0.5
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
            min_='0',
            max_='5',
            interval=0.5
        ),

    )
)
line=Line().set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(is_show=True),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),

        ),

    ).add_xaxis(xaxis_data=x_data33).set_series_opts(
        areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
        label_opts=opts.LabelOpts(is_show=False),
    ).set_series_opts(label_opts=opts.LabelOpts(is_show=False))


for x in x_data11:
    y = -1*x+4
    y_data11.append(y)
for x in x_data22:
    y = -1*x+2
    y_data22.append(y)
for x in x_data33:
    y = -1*x+6
    y_data33.append(y)


line.add_yaxis('',y_data11,label_opts=opts.LabelOpts(is_show=False),linestyle_opts=opts.LineStyleOpts(width=8,))
line.add_yaxis('',y_data22,label_opts=opts.LabelOpts(is_show=False),linestyle_opts=opts.LineStyleOpts(type_='dashed'),)
line.add_yaxis('',y_data33,label_opts=opts.LabelOpts(is_show=False),linestyle_opts=opts.LineStyleOpts(type_='dashed'),)
line.render('1.html')
line.overlap(scatter)
line.render('Example_final.html')


