reset
set style data histogram
set style fill solid
set term png enhanced font 'Verdana,10'
set output 'runtime.png'
set ylabel 'time ( us )'
set title 'multiplication algorighm runtime'

set xtics rotate by 30 offset -0.5,-1.5

plot [:][:]'runtime_us.txt' using 2:xtic(1) title 'runtime', \
'' using ($0):($2+0.01):2 with labels title ' '
