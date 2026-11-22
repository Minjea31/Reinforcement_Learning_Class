"""표시 수치는 소수점 3자리. 작은 설정값은 과학적 표기 유지."""
import matplotlib.ticker as mt

def display(v):
    if v and abs(v)<.001:return f'{v:.0e}'
    return f'{v:.3f}'
def format_figure(fig):
    fig.canvas.draw()
    for ax in fig.axes:
        for axis,label in [(ax.xaxis,ax.get_xlabel()),(ax.yaxis,ax.get_ylabel())]:
            if label.startswith('Updates') or label in ['Seed','Hidden nodes','Hidden layers']:continue
            if axis.get_scale()=='log':
                axis.set_major_locator(mt.LogLocator(base=10,subs=(1,2,5) if axis==ax.yaxis else (1,),numticks=15))
                axis.set_minor_formatter(mt.NullFormatter())
            formatter=axis.get_major_formatter()
            if isinstance(formatter,mt.FuncFormatter):
                ticks=axis.get_ticklocs();labels=[t.get_text() for t in axis.get_ticklabels()]
                try:vals=[float(t) for t in labels]
                except ValueError:continue
                axis.set_ticks(ticks,labels=[display(v) for v in vals])
            else:axis.set_major_formatter(mt.FuncFormatter(lambda v,pos:display(v)))
