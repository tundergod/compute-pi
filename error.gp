reset

#set logscale x 2
set grid
set key left
set xlabel 'N'
set ylabel 'error'
set style fill solid
set title 'error rate'
#set xrange[0:100000]
#set yrange[:]
set term png enhanced font 'Verdana,10'
set output 'error.png'

plot 'error.csv'\
   using 1:2 smooth csplines lw 1 title 'baseline error',\
'' using 1:3 smooth csplines lw 1 title 'leibniz error',\
