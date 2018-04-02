from bokeh.plotting import figure, show, output_file
import numpy

p = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p.line(y = volume['vol_24h'], x = volume.index)

output_file('ts.html')
show(p)


p2 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p2.line(y = vmc,x = volume.index)
output_file('ts2.html')
show(p2)

p3 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p3.line(y = volavg,x = volume.index)
output_file('ts3.html')
show(p3)

p4 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p4.line(y = vdiff,x = volume.index)
output_file('ts4.html')
show(p4)

p5 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p5.line(y = pusd,x = volume.index)
output_file('ts5.html')
show(p5)

tablelist = get_table_list(bqc,job_config)
syms = ['BTC']
sym = syms[0]
df = get_many_syms(syms,tablelist,bqc,job_config)
dset = df
volume = pd.DataFrame(dset.loc[sym].vol_24h)
vmc = dset.loc[sym].vol_24h / dset.loc[sym].mkt_cap
vmcdiff = vmc.diff()
pbtc = pd.DataFrame(dset.loc[sym].price_btc)
pbtc = pbtc.price_btc
pusd = pd.DataFrame(dset.loc[sym].price_usd)
pusd = pusd.price_usd
pdiff = numpy.log(1+pusd.pct_change())


'''
p2 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p2.line(y = vmc,x = volume.index)
output_file('vmc{}.html'.format(sym))
show(p2)

p3 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p3.line(y = pusd,x = volume.index)
output_file('pusd{}.html'.format(sym))
show(p3)

p5 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p5.line(y = pbtc,x = volume.index)
output_file('pbtc{}.html'.format(sym))
show(p5)

p6 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p6.line(y = vsum,x = volume.index)
output_file('v10{}.html'.format(sym))
show(p6)

p7 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p7.line(y = volume.vol_24h,x = volume.index)
output_file('v{}.html'.format(sym))
show(p7)
'''

p3 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
p3.line(y = pusd,x = volume.index)
output_file('pusd{}.html'.format(sym))
show(p3)

w = [10,20,50,100]

for i in w:
    vmcsum = vmcdiff.rolling(i).sum()
    p8 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
    p8.line(y = vmcsum,x = volume.index)
    output_file('vmcsum{}{}.html'.format(i,sym))
    show(p8)

w = [10,20,50,100]

for i in w:
    psum = pdiff.rolling(i).sum()
    p8 = figure(x_axis_type="datetime", plot_width = 800, plot_height=350)
    p8.line(y = psum,x = volume.index)
    output_file('psum{}{}.html'.format(i,sym))
    show(p8)