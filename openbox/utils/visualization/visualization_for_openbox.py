from pyecharts.charts import Page, Line, Scatter, Parallel
from pyecharts.components import Table
import pyecharts.options as opts


# task information table
def task_inf_table(data) -> Table:
    table = (
        Table().add(headers=data['table_field'], rows=[data['table_data']], attributes={
            "align": "r",
            "border": False,
            "style": "background:#FFFFFF; width:750px; height:80px; font-size:20px; color:#000000;"
        })
            .set_global_opts(
            title_opts=opts.ComponentTitleOpts(title="task information")
        )
    )

    return table


def per_line(data) -> Line:
    line = (
        Line()
            .add_xaxis([_[0] for _ in data['min']])
            .add_yaxis(
                series_name="折线图",

                symbol='circle',
                symbol_size=12,
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#4fbfa8"
                ),
                label_opts=opts.LabelOpts(is_show=False),

                y_axis=[_[1] for _ in data['min']],
                linestyle_opts=opts.LineStyleOpts(color="#4fbfa8"),

            )
            .set_global_opts(
            legend_opts=opts.LegendOpts(pos_left=10),
            xaxis_opts=opts.AxisOpts(
                type_="value",  # necessary
                name='Number of iterations n',
                boundary_gap=False,
                name_location='center',
                name_rotate=0,
                name_textstyle_opts=opts.TextStyleOpts(padding=[20, 0, 0, 0], font_size=16),
                splitline_opts=opts.SplitLineOpts(
                    is_show=True,
                    linestyle_opts=opts.LineStyleOpts(
                        color='black',
                        width=1,
                        type_='solid'
                    )
                )
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                name='Min objective value after n iterations',
                name_location='center',
                name_rotate=90,
                name_textstyle_opts=opts.TextStyleOpts(padding=[0, 0, 50, 50], font_size=16),
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(
                    is_show=True,
                    linestyle_opts=opts.LineStyleOpts(
                        color='black',
                        width=1,
                        type_='solid'
                    )
                ),
            ),
            tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='line'),
            datazoom_opts=[
                opts.DataZoomOpts(type_='slider', xaxis_index=0, range_start=0, range_end=100),
                opts.DataZoomOpts(type_='inside', xaxis_index=0, range_start=0, range_end=100)

            ]
        )
    )

    scatter1 = (
        Scatter()
            .add_xaxis([_[0] for _ in data['over']])
            .add_yaxis(
                series_name="散点图",
                y_axis=[_[1] for _ in data['over']],
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color='grey'),
                color='grey'
            )
    )
    line.overlap(scatter1)

    scatter2 = (
        Scatter()
            .add_xaxis([_[0] for _ in data['scat']])
            .add_yaxis(
                series_name="散点图",
                y_axis=[_[1] for _ in data['scat']],
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color='#4fbfa8'),
                color='#4fbfa8'
            )
    )
    line.overlap(scatter2)

    return line


def conf_parallel(data) -> Parallel:

    parallel = (
        Parallel()
            .add_schema([{"dim": i, "name": name} for i, name in enumerate(data['schema'])])
            .add(
                series_name='',
                data=data['data'],
                linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.75),
            )
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(
                min_=data['visualMap']['min'],
                max_=data['visualMap']['max'],
                dimension=data['visualMap']['dimension'],
                range_color=['#d94e5d','#eac736','#50a3ba'],
                pos_left='left'
            )
        )
    )

    return parallel


def vis_openbox(draw_data: dict, file_path: str):
    page = Page(interval=10, layout=Page.SimplePageLayout)

    page.add(
        task_inf_table(draw_data['task_inf']),
        per_line(draw_data['line_data']),
        conf_parallel(draw_data['parallel_data'])
    )
    page.render(file_path)
