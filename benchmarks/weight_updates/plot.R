
png('benchmarks.png', width = 1500, height = 1000)

d = read.csv('benchmarks.csv')
par(mfrow=c(2,4), cex=1)

for(i in c(1000, 10000, 100000, 1000000)) {
    peg = d[d$samples==i & d$model=='peg',]
    sgd = d[d$samples==i & d$model=='sgd',]

    if(nrow(peg) > 0 || nrow(sgd) > 0) {
        plot(peg$accuracy, type='l', col=3, yaxt='n', xaxt='n', ylim=c(0.6,1), 
             main=c(i,'samples'), ylab='accuracy', xlab='dataset weight updates')
        axis(2, c(0.6, 0.7, 0.8, 0.9, 1))
        axis(1, at=1:4, labels=c(1, 5, 10, 25))
        lines(sgd$accuracy, type='l', col=4)
        legend('bottom', col=c(4,3), c('sgd', 'pegasos'), lty=1)


    }
}

for(i in c(1000, 10000, 100000,1000000)) {
    peg = d[d$samples==i & d$model=='peg',]
    sgd = d[d$samples==i & d$model=='sgd',]

    if(nrow(peg) > 0 || nrow(sgd) > 0) {
        plot(sgd$time, type='l', col=4, xaxt='n', main=c(i,'samples'), 
             ylim=c(0,1000), ylab='time', xlab='dataset weight updates')
        axis(1, at=1:4, labels=c(1, 5, 10, 25))
        lines(peg$time, type='l', col=3)
        legend('top', col=c(4,3), c('sgd', 'pegasos'), lty=1)
    }
}

dev.off()
