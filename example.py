from sklearn.datasets import make_circles
import pyecharts.options as opts
from pyecharts.charts import Scatter,Scatter3D
import numpy as np
import pandas as pd

"""x1 , y1 = make_circles(n_samples=1000,factor=0.5,noise=0.1)
Alist = []
Blist = []
for i in range(len(y1)):
    if y1[i] == 0:
        Alist.append((list(x1[i])))
    else:
        Blist.append((list(x1[i])))
def cal_z(xydata,xyzdata):
    for i in range(len(xydata)):
        z=np.sqrt(xydata[i][0]*xydata[i][0]+xydata[i][1]*xydata[i][1])
        xydata[i].append(z)
        xyzdata.append(xydata[i])"""

"""A_xyz=[]
B_xyz=[]"""

"""cal_z(Alist,A_xyz)
cal_z(Blist,B_xyz)
fileA=pd.DataFrame(A_xyz)
fileA.to_csv('A.csv',index=None)
fileB=pd.DataFrame(B_xyz)
fileB.to_csv('B.csv',index=None)"""





"""
Alist=np.asarray(Alist)
Blist=np.asarray(Blist)
scatter=(
    Scatter(init_opts=opts.InitOpts(width="200px", height="200px"))
    .add_xaxis(xaxis_data=Alist[:,0])
    .add_yaxis(
        series_name="A",
        y_axis=Alist[:,1],
        symbol_size=5,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_xaxis(xaxis_data=Blist[:,0])
        .add_yaxis(
        series_name="B",
        y_axis=Blist[:,1],
        symbol_size=5,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_series_opts()
    .set_global_opts(
        title_opts=opts.TitleOpts(title="一个棘手的问题",pos_left="center"),
        xaxis_opts=opts.AxisOpts(
            type_="value", splitline_opts=opts.SplitLineOpts(is_show=True),

            interval=0.5
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
            interval=0.5
        ),
        legend_opts=opts.LegendOpts(pos_bottom='bottom',pos_left='center')
    )
)scatter.render('Hard-example.html')"""

"""
(
    Scatter3D(
        init_opts=opts.InitOpts(width="1440px", height="720px")
    )
    .add(
        series_name="A",
        data=A_xyz,
        xaxis3d_opts=opts.Axis3DOpts(
            name='x',
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        yaxis3d_opts=opts.Axis3DOpts(
            name='y',
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        zaxis3d_opts=opts.Axis3DOpts(
            name='z',
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        grid3d_opts=opts.Grid3DOpts(width=200, height=200, depth=200),

    )
.add(
        series_name="B",
        data=B_xyz,
        xaxis3d_opts=opts.Axis3DOpts(
            name='x',
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        yaxis3d_opts=opts.Axis3DOpts(
            name='y',
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        zaxis3d_opts=opts.Axis3DOpts(
            name='z',
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        grid3d_opts=opts.Grid3DOpts(width=200, height=200, depth=200),
    )
    .set_global_opts(
        visualmap_opts=[
            opts.VisualMapOpts(
                type_="size",
                is_calculable=True,
                dimension=4,
                pos_bottom="10",
                max_=2 / 2,
                range_size=[1, 5],
            ),
        ]
    )
    .render("scatter3d.html")
)
"""