# softbody-evolution
## **Modelling the growth of a soft-body system from a single cell and temporal adaptation to the environment**

This project presents the adaptation process of a soft-body system based on the spring interactions between the cells that make up the system, to the user-defined environment. The soft-body system is generated by consecutive mitosis a single-cell system whereas the environment includes desired and undesired regions. The cells undergo apoptosis under certain conditions: two cells overlapping each other, a cell that is not interacting with another cells, and the cells that positioned on the undesired regions. If none of the conditions for apoptosis is met, the cells that have interactions less than a predetermined value undergo mitosis and newborn cells are generated. Time integration is done by velocity-Verlet algorithm.

$x(t+\Delta t) = x(t) +v(t)\Delta t + \frac{1}{2}a(t)\Delta t^2$
$v(t+\Delta t) = v(t) + \frac{1}{2}[a(t)+a(t+\Delta t)] \Delta t$

This model is tested for varying cell diameter values and letters {U,N,A,M} (see [report](https://github.com/lynspica/softbody-evolution/blob/main/report.pdf)). 

|   Parameter   |     Value     |
| :---: | :---: |
| Cell diameter $(D_{cell})$ | 0.8, 0.85, 0.9, 0.95 1.0  |
| Spring constant $(k)$      | 0.5  |
| Cell mass $(m)$            | 1  |
| Timestep $(\delta t)$      | 0.1 |
| Max. number of interactions<br />to undergo mitosis  | 3 |
| Final system size      | $\geq$ 128 cells  |
| Interaction range      | 1.5 $(\times$ $D_{cell})$  |
| Interaction update frequency       | 10 timesteps  |
| Momentum loss in event:<br />*Collision* | 1 (elastic)  |
| Momentum loss in event:<br />*Division* | 1 (elastic)  |
| Total number of steps<br />in event: *Mitosis* | 50 timesteps  |
| Total number of steps<br />in event: *Apoptosis* | 25 timesteps  |

The snapshot below shows the drawing panel in which the user can provide the desired/undesired regions for the soft-body system to adapt. As an example, a symbol resembling the letter H is provided to the evolutionary process.

 <img src="https://github.com/lynspica/softbody-evolution/blob/main/figs/map.png" width="500" height="500">
 
The animation below shows the evolutionary process, including mitosis and apoptosis events. If # of neighboring cells (= # of interactions) are > 2, the cell can no longer perform mitosis and colored with red. If # of neighboring cells = 2, the cell can perform a single mitosis and colored with yellow. If # of neighboring cells < 2, the cell can perform mitosis and colored with green. 

![](https://github.com/lynspica/softbody-evolution/blob/main/figs/evolution.gif)

The snapshot below shows the final form of the soft-body adapted to the provided environment.

![](https://github.com/lynspica/softbody-evolution/blob/main/figs/final.png)
