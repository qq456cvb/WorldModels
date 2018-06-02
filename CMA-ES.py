import numpy as np


def cmaes(f, n, crit=1e-8, max_iter=1e5):
    xmean = np.random.rand(n)
    zmean = np.zeros(n)
    sigma = 0.5
    # set initial parameters
    lmbda = int(4 + np.floor(3 * np.log(n)))

    # only positive weights
    w = np.log(lmbda / 2 + 0.5) - np.log(np.arange(1, lmbda + 1))
    mu = lmbda // 2

    mueff = np.square(np.sum(w[:mu])) / np.sum(np.square(w[:mu]))
    mueff_neg = np.square(np.sum(w[mu:])) / np.sum(np.square(w[mu:]))

    cm = 1
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)

    alpha_cov = 2
    c1 = alpha_cov / ((n + 1.3) * (n + 1.3) + mueff)
    cmu = min(1 - c1, alpha_cov * (mueff - 2 + 1 / mueff) / ((n + 2) * (n + 2) + alpha_cov * mueff / 2))
    ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

    # normalize weight
    w[:mu] = w[:mu] / np.sum(w[:mu])
    alpha_mu_neg = 1 + c1 / cmu
    alpha_mueff_neg = 1 + 2 * mueff_neg / (mueff + 2)
    alpha_posdef_neg = (1 - c1 - cmu) / (n * cmu)

    w[mu:] = -min(min(alpha_mu_neg, alpha_mueff_neg), alpha_posdef_neg) * w[mu:] / np.sum(w[mu:])

    # start point
    pc = np.zeros(n)
    ps = np.zeros(n)
    B = np.eye(n)
    D = np.eye(n)
    C = np.eye(n)
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    eigeneval = 0
    for counteval in range(int(max_iter)):

        z = np.random.randn(n, lmbda)
        y = np.matmul(np.matmul(B, D), z)
        x = xmean[:, None] + sigma * y

        # sort by fitness
        fs = [f(x[:, j]) for j in range(lmbda)]
        best_idx = np.argsort(fs)

        xmean = (1 - cm) * xmean + cm * np.sum(w[None, :mu] * x[:, best_idx[:mu]], 1)
        zmean = (1 - cm) * zmean + cm * np.sum(w[None, :mu] * z[:, best_idx[:mu]], 1)

        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(B, zmean)
        sigma = sigma * np.exp((cs / ds) * (np.linalg.norm(ps) / chi_n - 1))

        hs = 1 if np.linalg.norm(ps) / np.sqrt(1 - np.power(1 - cs, 2 * (counteval + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * np.dot(np.matmul(B, D), zmean)

        wo = w.copy()
        wo[mu:] = wo[mu:] * n / np.square(np.linalg.norm(np.dot(B, z[:, best_idx]), axis=0))[mu:]
        C = (1 + c1 * (1 - hs) * cc * (2 - cc) - c1 - cmu * np.sum(w)) * C + c1 * np.outer(pc, pc) \
            + cmu * np.matmul(np.matmul(y[:, best_idx], np.diag(wo)), np.transpose(y[:, best_idx]))

        if counteval * lmbda - eigeneval > lmbda / (c1 + cmu) / n / 10:
            eigeneval = counteval
            C = np.triu(C) + np.transpose(np.triu(C, 1))
            dd, B = np.linalg.eig(C)
            D = np.diag(np.sqrt(dd))

        if fs[best_idx[0]] <= crit or counteval >= max_iter:
            print('found solution in %d iterations' % counteval)
            break

    print('best x: ', end='')
    print(x[:, best_idx[0]])
    print('best value %f' % f(x[:, best_idx[0]]))


if __name__ == '__main__':
    f = lambda x: np.dot(np.power(1e6, np.arange(x.shape[0]) / (x.shape[0] - 1)), np.square(x))
    cmaes(f, int(20))
