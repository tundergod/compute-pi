reset

set logscale x 2
set grid
set key left
set xlabel 'N'
set ylabel 'Time(sec)'
set style fill solid
set title 'Wall-clock time - using clockgettime()'
#set xrange[0:100000]
#set yrange[0:0.1]
set term png enhanced font 'Verdana,10'
set output 'runtime.png'

plot 'result_clock_gettime.csv'\
   using 1:2 smooth csplines lw 1 title 'baseline',\
'' using 1:7 smooth csplines lw 1 title 'leibniz', \
'' using 1:8 smooth csplines lw 1 title 'leibniz avx', \
'' using 1:9 smooth csplines lw 1 title 'leibniz avx unroll'
