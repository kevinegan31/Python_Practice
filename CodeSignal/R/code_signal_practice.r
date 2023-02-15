# CodeSignal Practice
# Ratiorg Statues
solution <- function(statues) {
    statues <- sort(statues)
    statues_min_max <- max(statues) - min(statues)
    statues_req <- statues_min_max - length(statues) + 1
    return(statues_req)
}
statues <- c(6, 2, 3, 8)
statues <- sort(statues)
statues_min_max <- max(statues) - min(statues)
statues_min_max - length(statues) + 1
solution(statues)

print("Hello World")
