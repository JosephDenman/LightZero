## Environment Versatility

- The following is a brief introduction to the environment supported by our zoo：

<details open><summary>Expand for full list</summary>

| No | Environment | Label | Visualization | Doc Links |
|:--:|:---------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|
1 | [board_games/tictactoe](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/tictactoe) |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |   ![original](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/tictactoe/tictactoe.gif)    |                                                               [env tutorial](https://en.wikipedia.org/wiki/Tic-tac-toe)                                                                |
|
2 |    [board_games/gomoku](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/gomoku)    |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |      ![original](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/gomoku/gomoku.gif)       |                                                                  [env tutorial](https://en.wikipedia.org/wiki/Gomoku)                                                                  |
|
3 |  [board_games/connect4](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/connect4)  |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |        ![original](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/connect4/connect4.gif)         |                                                                 [env tutorial](https://en.wikipedia.org/wiki/Connect4)                                                                 |
|
4 |             [game_2048](https://github.com/opendilab/LightZero/tree/main/zoo/game_2048)             |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |         ![original](https://github.com/opendilab/LightZero/tree/main/zoo/game_2048/game_2048.gif)          |                                                                   [env tutorial](https://en.wikipedia.org/wiki/2048)                                                                   |
|
5 |           [chess](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/chess)           |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |       ![original](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/chess/chess.gif)        |                                                                  [env tutorial](https://en.wikipedia.org/wiki/Chess)                                                                   |
|
6 |              [go](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/go)              |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |          ![original](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/go/go.gif)           |                                                                    [env tutorial](https://en.wikipedia.org/wiki/Go)                                                                    |
|
7 |  [classic_control/cartpole](https://github.com/opendilab/LightZero/tree/main/zoo/classic_control)   |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |                         ![original](./dizoo/classic_control/cartpole/cartpole.gif)                         |      [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/cartpole.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/cartpole_zh.html)      |
|
8 |  [classic_control/pendulum](https://github.com/opendilab/LightZero/tree/main/zoo/classic_control)   |                                 ![continuous](https://img.shields.io/badge/-continous-green)                                  | ![original](https://github.com/opendilab/DI-engine/blob/main//dizoo/classic_control/pendulum/pendulum.gif) |      [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/pendulum.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/pendulum_zh.html)      |
|
9 |           [box2d/lunarlander](https://github.com/opendilab/LightZero/tree/main/zoo/box2d)           | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  ![continuous](https://img.shields.io/badge/-continous-green) |   ![original](https://github.com/opendilab/DI-engine/blob/main//dizoo/box2d/lunarlander/lunarlander.gif)   |   [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/lunarlander.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/lunarlander_zh.html)   |
|
10 |          [box2d/bipedalwalker](https://github.com/opendilab/LightZero/tree/main/zoo/box2d)          |                                 ![continuous](https://img.shields.io/badge/-continous-green)                                  |   ![original](https://github.com/opendilab/DI-engine/blob/main//dizoo/box2d/bipedalwalker/bipedalwalker.gif)    | [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/bipedalwalker.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/bipedalwalker_zh.html) |
|
11 |                 [atari](https://github.com/opendilab/LightZero/tree/main/zoo/atari)                 |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |            ![original](https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/atari.gif)             |         [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/atari.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/atari_zh.html)         |
|
11 |                [mujoco](https://github.com/opendilab/LightZero/tree/main/zoo/mujoco)                |                                 ![continuous](https://img.shields.io/badge/-continous-green)                                  |           ![original](https://github.com/opendilab/DI-engine/blob/main/dizoo/mujoco/mujoco.gif)            |        [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/mujoco.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/mujoco_zh.html)        |
|
12 |              [minigrid](https://github.com/opendilab/LightZero/tree/main/zoo/minigrid)              |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |         ![original](https://github.com/opendilab/DI-engine/blob/main/dizoo/minigrid/minigrid.gif)          |      [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/minigrid.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/minigrid_zh.html)      |

</details>

![discrete](https://img.shields.io/badge/-discrete-brightgreen) means discrete action space

![continuous](https://img.shields.io/badge/-continous-green) means continuous action space

- Some environments, like the LunarLander, support both types of action spaces. For continuous action space environments
  such as BipedalWalker and Pendulum, you can manually discretize them to obtain discrete action spaces. Please refer
  to [action_discretization_env_wrapper.py](https://github.com/opendilab/LightZero/blob/main/lzero/envs/wrappers/action_discretization_env_wrapper.py)
  for more details.

- This list is continually updated as we add more game environments to our collection.