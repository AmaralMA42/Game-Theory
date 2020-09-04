import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
#Version 0.8
# Parâmetros da simulação:
L = 100  # tamanho da rede
amostras = 50
total_passos = 2000
passos_media = int(0.9 * total_passos)
k = 0.1     # Irracionalidade
R = [1, 1]  # Detalhe importante, cada rank dos payoffs se refere a uma cidade específica
P = [0, 0]  # Punição
T = [0.95, 1.04]  # Tentação
S = [0, 0]  # Sucker
p_mig = 1  # prob  de migração
n_mig = 100  # número de imigrantes por MCS
dens = 0.8  # densidade de preenchimento da rede # !!!frações de C e D em termos da densindade!!!
total_jog = L * L
start_time = time.time()
np.random.seed()


@jit(nopython=True)
def inicia_vizinhos(viz_):
    # Inicialização das estratégias e definição de vizinhos
    for jogador_atual in range(0, total_jog):  # Lembrete, for in range vai até total_jog-1!!
        viz_[jogador_atual, 1] = jogador_atual + 1  # Vizinho direito=1
        viz_[jogador_atual, 3] = jogador_atual - 1  # Vizinho esquerdo=3
        viz_[jogador_atual, 2] = jogador_atual + L  # Vizinho de baixo=2
        viz_[jogador_atual, 0] = jogador_atual - L  # Vizinho de cima=0
        # definindo condições de contorno periódicas na rede quadrada
        if (jogador_atual - L + 1) % L == 0:  # definindo coluna direita
            viz_[jogador_atual, 1] = jogador_atual + 1 - L  # definindo vizinho direito
        if jogador_atual % L == 0 or jogador_atual == 0:  # definindo coluna esquerda
            viz_[jogador_atual, 3] = jogador_atual + L - 1  # definindo vizinho esquerdo
        if L > jogador_atual >= 0:
            viz_[jogador_atual, 0] = jogador_atual + L * L - L
        if L * L > jogador_atual >= L * L - L:
            viz_[jogador_atual, 2] = jogador_atual - (L * L - L)


@jit(nopython=True)
def inicia_estrategias(estrategia_, rede):
    # Inicialização das estratégias
    temp_list = np.random.random(size=total_jog)  # Otimização, gera de uma vez lista de rands
    for jogador_atual in range(0, total_jog):  # Lembete, for in range vai até total_jog-1!!
        prob = temp_list[jogador_atual]  # Otimização, gera de uma vez lista de rands
        if prob < dens:  # probabilidade=dens do sítio ter agente em estado 0 ou 1
            estrategia_[rede][jogador_atual] = np.random.randint(2)  # Estado inicial do jogador como aleatório
        else:
            estrategia_[rede][jogador_atual] = 2  # estado de sítio vazio


@jit(nopython=True)  # todo Gasto de tempo enorme, como otimizar?
def calc_fracs(amo_, coops_t_, vazios_t_, estrategia_, passo_atual_):
    vazios_t_[0, amo_, passo_atual_] = np.sum(estrategia_[0] == 2) / total_jog  # Rede0
    vazios_t_[1, amo_, passo_atual_] = np.sum(estrategia_[1] == 2) / total_jog  # Rede1
    coops_t_[0, amo_, passo_atual_] = np.sum(estrategia_[0] == 0) / (total_jog * (1 - vazios_t_[0, amo_, passo_atual_]))
    coops_t_[1, amo_, passo_atual_] = np.sum(estrategia_[1] == 0) / (total_jog * (1 - vazios_t_[1, amo_, passo_atual_]))


@jit(nopython=True)
def prob_flip(pay_final, pay_ini):
    return 1 / (1 + np.exp(-(pay_final-pay_ini) / k))


@jit(nopython=True)
def calcula_payoff(sitio, rede, estrategia_, viz_, matriz_payoff_):
    payoff_sitio = 0
    est_sitio = estrategia_[rede][sitio]
    for j in range(0, 4):  # soma-se ao payoff o valor do jogo com vizinho para cada vizinho de 0 a 3
        payoff_sitio += matriz_payoff_[rede][est_sitio][estrategia_[rede][viz_[sitio, j]]]
    return payoff_sitio


@jit(nopython=True)
def atualiza_total_estrat(rede, estrategia_, viz_, matriz_payoff_):
    list_atual = np.random.randint(0, total_jog, size=total_jog)  # lista de jogadores aleatórios para um  MCS
    list_viz = np.random.randint(0, 4, size=total_jog)  # lista vizinhos (4) aleatórios a utilizar para um  MCS
    for cont in range(0, total_jog):
        atual = list_atual[cont]  # np.random.randint(0, total_jog)
        sorteio = list_viz[cont]  # np.random.randint(0, 4)  # sorteio de um vizinho aleatório de 0 a 3
        #  todo sortear somente da lista de ativos?
        vizsorteado = viz_[atual, sorteio]
        if estrategia_[rede][vizsorteado] != 2 and estrategia_[rede][atual] != 2:
            pay_atual = calcula_payoff(atual, rede, estrategia_, viz_, matriz_payoff_)
            pay_viz = calcula_payoff(vizsorteado, rede, estrategia_, viz_, matriz_payoff_)
            prob = np.random.rand()  # probabilidade aleatória
            chance_muda = prob_flip(pay_viz, pay_atual)  # Probabilidade de fermi
            if prob < chance_muda:
                estrategia_[rede][atual] = estrategia_[rede][vizsorteado]  # mudança da estratégia do sítio central


@jit(nopython=True)
def migracao(estrategia_, rede_a, rede_b):
    prob_list = np.random.random(size=n_mig)
    for cont in range(0, n_mig):
        # Otimização, gera de uma vez lista de rands   np.random.rand()
        if prob_list[cont] < p_mig:
            imig_a = np.random.randint(0, total_jog)
            imig_b = np.random.randint(0, total_jog)
            aux = estrategia_[rede_a][imig_a]
            estrategia_[rede_a][imig_a] = estrategia_[rede_b][imig_b]  # mudança da estratégia do sítio central
            estrategia_[rede_b][imig_b] = aux  # mudança da estratégia do sítio central


@jit(nopython=True)
def monte_carlo(amo_, viz_, matriz_payoff_, estrategia_, coops_t, vazios_t_):
    # Inicio das estratégias para amostra
    inicia_estrategias(estrategia_, 0)
    inicia_estrategias(estrategia_, 1)
    # activ_list = [i for i in len(estrategia) if estrategia[i] != 2]  # lista de sítios ativos
    # Começo da simulação de Monte-Carlo
    for passo_atual in range(0, total_passos):
        # Antes de qualquer alteração, calcular todas estatísticas da população no passo atual
        calc_fracs(amo_, coops_t, vazios_t_, estrategia_, passo_atual)
        # Etapa de atualização da estratégia
        atualiza_total_estrat(0, estrategia_, viz_, matriz_payoff_)  # atualiza cada rede individualmente
        atualiza_total_estrat(1, estrategia_, viz_, matriz_payoff_)  # atualiza cada rede individualmente
        # Etapa de migração
        migracao(estrategia_, 0, 1)


# @jit(nopython=True) Numba não compatível com impressão aparentemente
def imprime_dados(coop_medio_t, v_medio_t):
    arquivo_escrito = open("cooperadores.dat", "w+")
    for cont in range(0, total_passos):
        arquivo_escrito.write(f"{cont} {coop_medio_t[0, cont]:.4f} {coop_medio_t[1, cont]:.4f} "
                              f"{v_medio_t[0, cont]:.4f} {v_medio_t[1, cont]:.4f} \n")
    for rede in range(0, 2):
        rhoT = np.average(coop_medio_t[rede, passos_media:total_passos])  # Média estavel de rho p/ todas amostras
        sigmaT = np.std(coop_medio_t[rede, passos_media:total_passos])  # desvio padrão destes
        densy = 1 - np.average(v_medio_t[rede, passos_media:total_passos])
        print(f"<C{rede}>= {rhoT:.4f} +- {sigmaT:.4f}  density{rede} = {densy:.4f} ")
    print(f"----- {(time.time() - start_time):.4f} seconds or {((time.time() - start_time) / 60.0):.4f} minutes-----")
    arquivo_escrito.close()


def plota_dados(coop_t, vaz_t, coop_medio_t, vaz_medio_t):
    fig, ax = plt.subplots(2, 1)  # Figure 1
    plt.suptitle('Cooperadores por amostra')
    plt.xlabel('Número de Passos')
    for i_ in range(0, 2):
        for amo_ in range(amostras):
            ax[i_].plot(coop_t[i_, amo_, :])
        ax[i_].set_ylabel(r'$\rho$')  # r transforma a string e mraw, permitindo comandos latex

    fig, ax = plt.subplots(2, 1)    # Figure 2
    plt.suptitle('Evolução média da cooperação')
    plt.xlabel('Número de Passos')
    for i_ in range(0, 2):
        ax[i_].set_ylabel(r'$<\rho >$')
        ax[i_].plot(coop_medio_t[i_, :])
        ax[i_].plot(vaz_medio_t[i_, :])

    fig, ax = plt.subplots(2, 1)    # Figure 3
    plt.suptitle('Histograma de distribuição final de cooperadores')
    plt.xlabel(r'$\rho$')
    # plt.xlim(-0.1, 1.1)
    for i_ in range(0, 2):
        ax[i_].hist(coop_medio_t[i_, passos_media:total_passos])
        ax[i_].set_ylabel(r'$P(\rho)$')
    plt.show()


# Definição de variáveis do jogo
estrategia = np.zeros((2, total_jog), dtype=int)
cooperadores_t = np.zeros((2, amostras, total_passos))
vazios_t = np.zeros((2, amostras, total_passos))
matriz_payoff = np.zeros((2, 3, 3))   # rede,linha,coluna
# Definição de vizinhos
viz = np.zeros((total_jog, 4), dtype=int)  # matriz contendo os vizinhos 0=cima,1=direita,2=baixo,3=esquerda

# 0 Copera 1 Deserta
for i in range(0, 2):  # Detalhe importante, cada rank inicial dos payoffs se refere a uma cidade específica
    matriz_payoff[i][0][0] = R[i]  # Recompensa
    matriz_payoff[i][1][0] = T[i]  # Tentação
    matriz_payoff[i][1][1] = P[i]  # Punição
    matriz_payoff[i][0][1] = S[i]  # Sucker

# ************************************** início*************************************************
inicia_vizinhos(viz)
# todo fazer a amostragem virar uma função que altera o vetor estrat, utilizar multiprocessing nela
for amo in range(0, amostras):  # loop de amostras
    # Monte-Carlo para uma amostra com "total_passos" passos de tempo
    monte_carlo(amo, viz, matriz_payoff, estrategia, cooperadores_t, vazios_t)

cooperadores_medios_t = np.sum(cooperadores_t, axis=1) / amostras
vazios_medios_t = np.sum(vazios_t, axis=1) / amostras
plota_dados(cooperadores_t, vazios_t, cooperadores_medios_t, vazios_medios_t)
imprime_dados(cooperadores_medios_t, vazios_medios_t)
