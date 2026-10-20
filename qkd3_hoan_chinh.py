# -*- coding: utf-8 -*-
"""
qkd3_hoan_chinh.py - QKD-3: bo dieu khien LUT hysteresis cho DV-QKD/FSO (phan mo phong Python).

MUC TIEU: tim hai cap nguong (nguong len, nguong xuong) cho HAI nhom kenh, roi xuat LUT cho MCU.
  - Nhom "yeu"     : nhieu yeu, tin hieu it dao dong.
  - Nhom "vua-manh": gop hai muc "vua" va "manh" vi chi so nhap nhay (SI) khong phan biet duoc.

CAU TRUC FILE (moi KHOI lam MOT viec):
  KHOI 0 : Tham so chung va CAP NGUONG DA CHOT (CAP_CHOT)
  KHOI 1 : Sinh du lieu kenh Gamma-Gamma (doc lap va co tuong quan)
  KHOI 2 : Bo dieu khien: nguong don, hysteresis, dem so lan doi
  KHOI 3 : Cac thuoc do: sai che do, tre xuong, ham chi phi J
  KHOI 4 : Du lieu theo nhom nhieu (gop vua + manh)
  KHOI 5 : Tim cap nguong bang tim luoi (lap nhieu lan, lay trung binh)
  KHOI 6 : Danh gia cap nguong tren chuoi MOI, so voi nguong don va cap cu
  KHOI 7 : LUT hai muc va xuat file lut.h cho firmware
  KHOI 8 : Chon nhom bang chi so nhap nhay SI
  KHOI 9 : Demo minh hoa (nhieu chuan; nguong co dinh khong hop moi muc)
  KHOI 10: main - bat/tat tung giai doan o dict CHAY

LUU Y: moi giai doan tu dat lai seed nen ket qua tung giai doan nhat quan, khong phu thuoc
giai doan chay truoc.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma as gm


# =====================================================================
# KHOI 0: THAM SO CHUNG VA CAP NGUONG DA CHOT
# Moi hang so nam o day. Khong de "so cung" rai rac trong code.
# =====================================================================
SEED = 1                 # hat giong so ngau nhien (de ket qua lap lai duoc)
N = 500                  # so mau moi chuoi
RHO = 0.9                # he so tuong quan giua hai mau lien tiep (kenh "nho")

# Ba muc nhieu goc: ten -> (alpha, beta) cua kenh Gamma-Gamma.
# alpha, beta cang nho thi nhieu cang manh. Gia tri chi de bat dau, can tra lai tai lieu.
MUC = {"yếu": (11, 10), "vừa": (4, 2), "mạnh": (4.2, 1.4)}

# Hai nhom dung de chon bang LUT. Moi nhom la danh sach cac muc goc nam trong no.
NHOM = {"yếu": ["yếu"], "vừa-mạnh": ["vừa", "mạnh"]}

# Che do ly tuong = trung binh truot cua tin hieu so voi nguong danh nghia T_DON.
CUA_SO_LY_TUONG = 25
BIEN = CUA_SO_LY_TUONG // 2   # bo ngan nay mau o moi dau chuoi khi tinh ty le sai (loi bien cua convolve)

# Buoc nhay cua kenh khi do tre: he so nhan 1.3 (kenh tot) roi 0.5 (kenh xau) o giua chuoi.
HE_SO_TOT, HE_SO_XAU = 1.3, 0.5
TRE_MAX = 50             # chan do tre toi da 50 mau (de vai chuoi "ket" khong lam meo trung binh)

# Ham chi phi J = W_DOI*doi/N + W_SAI*sai + W_TRE*tre/TRE_MAX
W_DOI, W_SAI = 1.0, 1.0
W_CHON = 1.0             # W_TRE dung de chon cap cuoi (quet thu 0.3, 1.0, 2.0 o DANH_SACH_W)
DANH_SACH_W = (0.3, 1.0, 2.0)

# Luoi tim nguong: nguong = he_so * T_DON
HE_SO_XUONG = np.linspace(0.3, 1.0, 15)   # nguong xuong
HE_SO_LEN = np.linspace(1.0, 2.4, 15)     # nguong len

# Cach tim: lap SO_LAN_LAP lan, moi lan SO_CHUOI_MOI_LAN chuoi MOI cho moi nhom
SO_LAN_LAP = 3
SO_CHUOI_MOI_LAN = 50
SO_CHUOI_KT = 300        # so chuoi MOI dung de kiem tra, cho TUNG muc goc

# Luong tu hoa cho LUT: tin hieu thuc 0..QMAX -> so nguyen 0..NBIN-1
QMAX, NBIN = 3.0, 256
BUOC = QMAX / (NBIN - 1)

# >>> HAI CAP NGUONG DA CHOT (nguong len, nguong xuong), cung thang do voi q (T_DON ~ 0.73) <<<
# Cach chot (W_TRE = 1.0): tim luoi 10 lan x 50 chuoi voi 2 seed khac nhau, roi so cac ung vien
# tren 1000 chuoi MOI cho moi muc goc. Tieu chi: giam so lan doi con >= 60% o CA "vua" va "manh"
# (co du dia so voi chi tieu >= 50%), tre xuong khong qua vai mau, sai che do khong te hon nguong don.
#   "yeu"     : ket qua tim luoi dao dong 0.80-0.87 / 0.47-0.55; chon 0.80 / 0.55
#               (giam ~65%, sai ~13%, tre xuong ~2.8 mau; cap 0.80/0.51 giam nhieu hon nhung tre 3.8 mau)
#   "vua-manh": ket qua tim luoi dao dong xuong 0.40-0.55; chon diem GIUA la 1.24 / 0.50
#               (giam ~65% o "vua", ~62% o "manh"; tre xuong ~3.0 / 2.3 mau)
# Cac con so nay phu thuoc kich ban mo phong (kenh, buoc nhay, rho): doi kich ban thi chay lai KHOI 5.
CAP_CHOT = {"yếu": (0.80, 0.55), "vừa-mạnh": (1.24, 0.50)}

# Cac cap dung o cac ban truoc, chi de so sanh trong KHOI 6
CAP_THAM_CHIEU = {
    "yếu":  {"cũ": (0.80, 0.51)},
    "vừa":  {"riêng cũ": (1.24, 0.36), "chung cũ": (1.24, 0.45)},
    "mạnh": {"riêng cũ": (1.24, 0.55), "chung cũ": (1.24, 0.45)},
}

VE_BIEU_DO = True        # dat False neu khong muon cua so bieu do chan chuong trinh


# =====================================================================
# KHOI 1: SINH DU LIEU KENH
# Tin hieu q[k] la cuong do chuan hoa (trung binh xap xi 1).
# =====================================================================
def gamma_gamma(alpha, beta, n):
    """n mau Gamma-Gamma DOC LAP: tich cua hai bien Gamma doc lap, moi bien co trung binh 1."""
    x = np.random.gamma(alpha, 1 / alpha, n)
    y = np.random.gamma(beta, 1 / beta, n)
    return x * y


def ar1_gauss(n, rho):
    """Chuoi Gauss co 'nho': mau sau phu thuoc mau truoc qua he so rho (qua trinh AR(1))."""
    z = np.zeros(n)
    z[0] = np.random.normal()
    for i in range(1, n):
        z[i] = rho * z[i - 1] + np.sqrt(1 - rho ** 2) * np.random.normal()
    return z


def gamma_tq(alpha, n, rho):
    """Chuoi Gamma CO TUONG QUAN: Gauss co nho -> phan bo deu (norm.cdf) -> phan bo Gamma (gm.ppf).
    Phan bo bien van la Gamma, tinh 'nho' van duoc giu."""
    u = norm.cdf(ar1_gauss(n, rho))
    return gm.ppf(u, a=alpha, scale=1 / alpha)


def gamma_gamma_tq(alpha, beta, n, rho):
    """Kenh Gamma-Gamma co tuong quan theo thoi gian (gan voi kenh khi quyen that)."""
    return gamma_tq(alpha, n, rho) * gamma_tq(beta, n, rho)


def tinh_t_don():
    """Nguong don danh nghia T_DON = trung vi cua muc 'vua'. Moi nguong khac tinh theo he so x T_DON."""
    a, b = MUC["vừa"]
    return np.percentile(gamma_gamma(a, b, 200000), 50)


np.random.seed(SEED)
T_DON = tinh_t_don()


# =====================================================================
# KHOI 2: BO DIEU KHIEN
# =====================================================================
def nguong_don(q, nguong):
    """Nguong DON: q >= nguong -> che do 1 (toc do cao), nguoc lai -> che do 0 (bao thu).
    Day la duong chuan de chung minh hysteresis co loi."""
    return [1 if x >= nguong else 0 for x in q]


def hysteresis(q, nguong_len, nguong_xuong, hien_tai=0):
    """HYSTERESIS = may trang thai 2 che do voi HAI nguong rieng:
         dang o 0: len 1 khi q >= nguong_len
         dang o 1: xuong 0 khi q <  nguong_xuong
       Vung giua hai nguong la 'vung tre': o do che do KHONG doi, nen nhieu quanh nguong
       khong gay dao dong. Dat nguong_len cao, nguong_xuong sat tam => len than trong, xuong nhanh.
       hien_tai: che do luc bat dau (0 hoac 1)."""
    che_do = []
    for x in q:
        if hien_tai == 0 and x >= nguong_len:
            hien_tai = 1
        elif hien_tai == 1 and x < nguong_xuong:
            hien_tai = 0
        che_do.append(hien_tai)
    return che_do


def dem_chuyen(che_do):
    """So lan che do KHAC mau truoc = so lan chuyen che do."""
    return sum(1 for i in range(1, len(che_do)) if che_do[i] != che_do[i - 1])


# =====================================================================
# KHOI 3: CAC THUOC DO
# Chi dem so lan doi la khong du: bo dieu khien khong bao gio doi thi dem = 0 nhung vo dung.
# Nen can them 'sai che do' va 'tre xuong'.
# =====================================================================
def che_do_ly_tuong(q, nguong=None, cua_so=CUA_SO_LY_TUONG):
    """Che do 'dung' de so sanh: lam muot tin hieu bang trung binh truot roi so voi nguong.
    Dinh nghia nay anh huong ket qua 'sai che do', can ghi ro trong bao cao."""
    nguong = T_DON if nguong is None else nguong
    tb = np.convolve(q, np.ones(cua_so) / cua_so, mode="same")
    return (tb >= nguong).astype(int)


def ty_le_sai(che_do, ly_tuong):
    """Ti le mau sai che do, BO 'BIEN' mau o moi dau chuoi.
    Ly do: np.convolve(mode='same') dem so 0 o hai dau, nen che do ly tuong o do khong dang tin."""
    return np.mean(np.array(che_do)[BIEN:-BIEN] != ly_tuong[BIEN:-BIEN])


def tao_tap_nhay(a, b, so_chuoi):
    """Chuoi kenh co BUOC NHAY that: nua dau tot (x1.3), nua sau xau (x0.5). Dung de do tre xuong."""
    he_so = np.concatenate([np.full(N // 2, HE_SO_TOT), np.full(N - N // 2, HE_SO_XAU)])
    return [gamma_gamma_tq(a, b, N, RHO) * he_so for _ in range(so_chuoi)]


def tre_xuong(chuoi_nhay, nl, nx):
    """Do tre xuong trung binh (so mau): sau buoc nhay kenh xau di, bo dieu khien (dang o che do 1)
    mat bao nhieu mau moi ve che do 0. Chan o TRE_MAX."""
    tre = []
    for q in chuoi_nhay:
        sau = q[N // 2:]
        cd = hysteresis(sau, nl, nx, hien_tai=1)
        t = next((i for i, c in enumerate(cd) if c == 0), TRE_MAX)
        tre.append(min(t, TRE_MAX))
    return np.mean(tre)


def chi_phi_goc(tap, nl, nx):
    """Tra ve (J_goc, so lan doi TB, ti le sai TB) cua cap (nl, nx) tren tap chuoi.
    J_goc = W_DOI*doi/N + W_SAI*sai. So hang tre xuong cong sau (xem KHOI 5), vi trong so
    W_TRE doi duoc ma khong phai tinh lai J_goc."""
    chuoi, ly_tuong = tap
    doi, sai = [], []
    for q, lt in zip(chuoi, ly_tuong):
        cd = hysteresis(q, nl, nx)
        doi.append(dem_chuyen(cd) / N)
        sai.append(ty_le_sai(cd, lt))
    return W_DOI * np.mean(doi) + W_SAI * np.mean(sai), np.mean(doi) * N, np.mean(sai)


# =====================================================================
# KHOI 4: DU LIEU THEO NHOM NHIEU
# Nhom 'vua-manh' lay mot nua chuoi tu 'vua', mot nua tu 'manh'.
# =====================================================================
def tao_tap_muc(ten, so_chuoi):
    """Tao 'so_chuoi' chuoi kenh cua mot muc goc, kem che do ly tuong."""
    a, b = MUC[ten]
    chuoi = [gamma_gamma_tq(a, b, N, RHO) for _ in range(so_chuoi)]
    return chuoi, [che_do_ly_tuong(q) for q in chuoi]


def tao_tap_nhom(nhom, so_chuoi):
    """Tap chuoi cua mot nhom, chia deu cho cac muc goc trong nhom."""
    cons = NHOM[nhom]
    moi_muc = max(1, so_chuoi // len(cons))
    chuoi, ly_tuong = [], []
    for ten in cons:
        c, l = tao_tap_muc(ten, moi_muc)
        chuoi += c
        ly_tuong += l
    return chuoi, ly_tuong


def tao_tap_nhay_nhom(nhom, so_chuoi):
    """Chuoi co buoc nhay cua mot nhom, dung de do tre xuong."""
    cons = NHOM[nhom]
    moi_muc = max(1, so_chuoi // len(cons))
    ds = []
    for ten in cons:
        ds += tao_tap_nhay(*MUC[ten], moi_muc)
    return ds


# =====================================================================
# KHOI 5: TIM CAP NGUONG BANG TIM LUOI
# Y tuong: thu moi cap (len, xuong) trong luoi, chon cap co J nho nhat. De giam nhieu khi
# "chon nguoi tot nhat trong 225 cap", ta LAP nhieu lan tren cac tap chuoi MOI va lay J trung binh.
# =====================================================================
def lap_bang_nhom(nhom, so_chuoi):
    """Voi MOI cap trong luoi: tinh J_goc va tre xuong tren mot tap chuoi MOI.
    Tra ve dict {(kl, kx): (J_goc, tre)}."""
    tap = tao_tap_nhom(nhom, so_chuoi)
    nhay = tao_tap_nhay_nhom(nhom, so_chuoi)
    bang = {}
    for kx in HE_SO_XUONG:
        for kl in HE_SO_LEN:
            J, _, _ = chi_phi_goc(tap, kl * T_DON, kx * T_DON)
            bang[(kl, kx)] = (J, tre_xuong(nhay, kl * T_DON, kx * T_DON))
    return bang


def chon_tu_bang(cac_bang, w_tre):
    """Chon cap co J trung binh (qua cac bang) nho nhat, J = J_goc + w_tre*tre/TRE_MAX."""
    tot = None
    for khoa in cac_bang[0]:
        J = np.mean([b[khoa][0] for b in cac_bang]) \
            + w_tre * np.mean([b[khoa][1] for b in cac_bang]) / TRE_MAX
        if tot is None or J < tot[0]:
            tot = (J, khoa)
    kl, kx = tot[1]
    return kl * T_DON, kx * T_DON


def tim_cap_cac_nhom():
    """Voi tung nhom: lap SO_LAN_LAP lan, in cap tot nhat cua tung lan (de thay do on dinh),
    roi chon cap cuoi tu TRUNG BINH. Tra ve {(nhom, w): (len, xuong)}."""
    cac_cap = {}
    for nhom in NHOM:
        print("--- Tìm cặp ngưỡng cho nhóm:", nhom, "(%d lần x %d chuỗi) ---"
              % (SO_LAN_LAP, SO_CHUOI_MOI_LAN))
        cac_bang = []
        for lan in range(SO_LAN_LAP):
            cac_bang.append(lap_bang_nhom(nhom, SO_CHUOI_MOI_LAN))
            nl, nx = chon_tu_bang([cac_bang[-1]], W_CHON)
            print("  lần", lan + 1, "| cặp chọn riêng lần này (W_TRE = %.1f):" % W_CHON,
                  round(nl, 2), "/", round(nx, 2))
        for w in DANH_SACH_W:
            cac_cap[(nhom, w)] = chon_tu_bang(cac_bang, w)
            nl, nx = cac_cap[(nhom, w)]
            print("  cặp cuối | W_TRE =", w, "| lên =", round(nl, 2), "xuống =", round(nx, 2))
    return cac_cap


# =====================================================================
# KHOI 6: DANH GIA TREN CHUOI MOI
# Cap nguong phai duoc danh gia tren chuoi KHAC voi chuoi dung de tim, de tranh chon trung cap 'may man'.
# =====================================================================
def danh_gia_cac_cap(ten, cac_cap, so_chuoi=SO_CHUOI_KT):
    """Danh gia nhieu cap nguong tren CUNG mot tap chuoi moi cua muc goc 'ten'.
    Tra ve {ten_cap: (doi, sai, tre_tb, tre_se)}."""
    a, b = MUC[ten]
    chuoi = [gamma_gamma_tq(a, b, N, RHO) for _ in range(so_chuoi)]
    ly = [che_do_ly_tuong(q) for q in chuoi]
    nhay = tao_tap_nhay(a, b, so_chuoi)
    kq = {}
    for ten_cap, (nl, nx) in cac_cap.items():
        _, doi, sai = chi_phi_goc((chuoi, ly), nl, nx)
        tre = np.array([tre_xuong([q], nl, nx) for q in nhay])
        kq[ten_cap] = (doi, sai, tre.mean(), np.std(tre) / np.sqrt(len(tre)))
    return kq


def danh_gia_ket_qua(cap_chot, cac_cap_them=None):
    """Voi tung muc goc: so nguong don, cap cu, cap CHOT (va cac cap moi tim neu co)."""
    print("--- Đánh giá trên chuỗi mới (%d chuỗi mỗi mức) ---" % SO_CHUOI_KT)
    for nhom, cons in NHOM.items():
        for ten in cons:
            cac = {"đơn": (T_DON, T_DON)}
            cac.update(CAP_THAM_CHIEU[ten])
            if cac_cap_them:
                for w in DANH_SACH_W:
                    cac["tìm W=%.1f" % w] = cac_cap_them[(nhom, w)]
            cac["CHỐT"] = cap_chot[nhom]
            kq = danh_gia_cac_cap(ten, cac)
            doi_don = kq["đơn"][0]
            print("[", ten, "] nhóm", nhom)
            for ten_cap, (doi, sai, tre, se) in kq.items():
                giam = (doi_don - doi) / doi_don * 100
                nl, nx = cac[ten_cap]
                print("   %-12s lên %.2f xuống %.2f | giảm %3.0f%% | sai %2.0f%% | trễ xuống %.1f ± %.1f mẫu"
                      % (ten_cap, nl, nx, giam, 100 * sai, tre, se))


# =====================================================================
# KHOI 7: BANG TRA LUT (dang MCU se chay) VA FILE lut.h
# Tren MCU khong so sanh so thuc. Thay vao do: luong tu hoa q thanh bin 0..255, roi tra bang
# LUT[nhom][che_do_hien_tai][bin] -> che_do_ke_tiep.
# =====================================================================
def luong_tu_hoa(q):
    """Doi tin hieu thuc thanh so nguyen 0..NBIN-1 (viec dau tien MCU lam voi moi mau)."""
    return np.clip(np.round(np.array(q) / BUOC), 0, NBIN - 1).astype(np.uint8)


def tao_lut(nl, nx):
    """Dien bang 2 x NBIN tu hai nguong:
         lut[0][bin] = 1 neu bin >= bin_nguong_len   (dang o 0: len khi du cao)
         lut[1][bin] = 0 neu bin <  bin_nguong_xuong (dang o 1: xuong khi qua thap)"""
    bl, bx = int(round(nl / BUOC)), int(round(nx / BUOC))
    lut = np.zeros((2, NBIN), dtype=np.uint8)
    for b in range(NBIN):
        lut[0][b] = 1 if b >= bl else 0
        lut[1][b] = 0 if b < bx else 1
    return lut


def chay_lut(q, lut):
    """Toan bo bo dieu khien chi la mot phep tra bang cho moi mau: m = lut[m][bin]."""
    m, ra = 0, []
    for b in luong_tu_hoa(q):
        m = int(lut[m][b])
        ra.append(m)
    return ra


def tao_lut_hai_muc(cap):
    """Hai bang LUT (moi bang 2 x NBIN byte) tu hai cap nguong cua hai nhom."""
    return {nhom: tao_lut(*cap[nhom]) for nhom in NHOM}


def so_sanh_lut(lut_hai, cap, so_chuoi=100):
    """Kiem tra LUT cho ket qua gan trung ban Python (lech nho chi do lam tron khi luong tu hoa)."""
    print("--- So sánh LUT với bản Python ---")
    for nhom, cons in NHOM.items():
        nl, nx = cap[nhom]
        for ten in cons:
            a, b = MUC[ten]
            lech, doi_py, doi_lut = [], [], []
            for _ in range(so_chuoi):
                q = gamma_gamma_tq(a, b, N, RHO)
                cd_py, cd_lut = hysteresis(q, nl, nx), chay_lut(q, lut_hai[nhom])
                lech.append(np.mean(np.array(cd_py) != np.array(cd_lut)))
                doi_py.append(dem_chuyen(cd_py))
                doi_lut.append(dem_chuyen(cd_lut))
            print(ten, "| bảng nhóm", nhom, "| lệch:", round(100 * np.mean(lech), 2), "% mẫu",
                  "| đổi (Python/LUT):", round(np.mean(doi_py), 1), "/", round(np.mean(doi_lut), 1))


def ghi_lut_h(lut_hai, cap, ten_file="lut.h"):
    """Xuat 2 bang ra file C cho firmware STM32. Chu thich khong dau de tranh loi bang ma tren Windows.
    Ghi them so bin cua tung nguong de firmware doi chieu nhanh."""
    with open(ten_file, "w", encoding="utf-8") as f:
        f.write("#include <stdint.h>\n#define NBIN %d\n" % NBIN)
        f.write("// luong tu hoa: bin = round(q * %d / %.1f), cat o 0..%d\n" % (NBIN - 1, QMAX, NBIN - 1))
        for i, nhom in enumerate(NHOM):
            nl, nx = cap[nhom]
            f.write("// nhom %d = %s : nguong len bin %d, nguong xuong bin %d\n"
                    % (i, nhom.replace("ế", "e").replace("ừ", "u").replace("ạ", "a"),
                       int(round(nl / BUOC)), int(round(nx / BUOC))))
        f.write("// chi so: [nhom nhieu: 0=yeu,1=vua-manh][che do hien tai][bin]\n")
        f.write("const uint8_t LUT[2][2][NBIN] = {\n")
        for nhom in NHOM:
            f.write("  {\n")
            for m in range(2):
                f.write("    {" + ",".join(str(int(v)) for v in lut_hai[nhom][m]) + "},\n")
            f.write("  },\n")
        f.write("};\n")
    print("Đã ghi", ten_file, "| kích thước dữ liệu:", 2 * 2 * NBIN, "byte")


# =====================================================================
# KHOI 8: CHON NHOM BANG CHI SO NHAP NHAY SI
# MCU do SI = phuong sai / trung binh^2 tren cua so W mau gan nhat, so voi mot ranh gioi de chon bang.
# Ranh gioi dat giua p95 cua 'yeu' va p5 cua 'vua-manh' de hai loai loi can bang.
# =====================================================================
def chi_so_nhap_nhay(q):
    """SI = phuong sai / trung binh^2. Nhieu manh thi SI lon."""
    return np.var(q) / np.mean(q) ** 2


def do_si(a, b, w, so_lan=300):
    """SI cua 'so_lan' cua so do dai w mau, de xem phan bo SI cua tung muc nhieu."""
    return np.array([chi_so_nhap_nhay(gamma_gamma_tq(a, b, w, RHO)) for _ in range(so_lan)])


def phan_loai_si(cac_w=(100, 200, 400), so_lan=300):
    print("--- Chọn nhóm bằng SI ---")
    for w in cac_w:
        si_yeu = do_si(*MUC["yếu"], w, so_lan)
        si_vua = do_si(*MUC["vừa"], w, so_lan)
        si_manh = do_si(*MUC["mạnh"], w, so_lan)
        si_vm = np.concatenate([si_vua, si_manh])
        ranh = (np.percentile(si_yeu, 95) + np.percentile(si_vm, 5)) / 2
        print("W =", w, "| ranh giới SI =", round(ranh, 3),
              "| yếu đoán đúng:", round(100 * np.mean(si_yeu < ranh)), "%",
              "| vừa đoán đúng:", round(100 * np.mean(si_vua >= ranh)), "%",
              "| mạnh đoán đúng:", round(100 * np.mean(si_manh >= ranh)), "%")


# =====================================================================
# KHOI 9: DEMO MINH HOA (phuc vu bao cao, khong anh huong cap nguong)
# =====================================================================
def demo_co_ban():
    """Tin hieu dao dong quanh 5: so nguong don voi hysteresis, roi quet be rong vung tre.
    Dong 'vung tre 0.0' phai trung nguong don (de kiem tra code dung)."""
    q = 5 + np.random.normal(0, 0.5, N)
    d_don = dem_chuyen(nguong_don(q, 5.0))
    d_hys = dem_chuyen(hysteresis(q, 5.5, 4.5))
    print("--- Demo cơ bản ---")
    print("Ngưỡng đơn:", d_don, "lần đổi | Hysteresis:", d_hys, "lần đổi | giảm:",
          round((d_don - d_hys) / d_don * 100, 1), "%")
    rong = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    ket_qua = [dem_chuyen(hysteresis(q, 5 + h / 2, 5 - h / 2)) for h in rong]
    for h, k in zip(rong, ket_qua):
        print("Vùng trễ", h, "->", k, "lần đổi")
    if VE_BIEU_DO:
        plt.plot(q)
        plt.axhline(5, color="red")
        plt.title("Tín hiệu giả lập quanh ngưỡng 5")
        plt.show()
        plt.figure()
        plt.plot(rong, ket_qua, "o-")
        plt.xlabel("Bề rộng vùng trễ")
        plt.ylabel("Số lần đổi chế độ")
        plt.show()


def demo_nguong_co_dinh():
    """Ap MOT bo nguong co dinh (tinh tu muc 'vua') cho ca ba muc nhieu.
    Cho thay nguong co dinh khong hop moi muc: muc 'yeu' bi ket o che do 0 (sai che do rat cao).
    Day la ly do can nguong rieng cho tung nhom."""
    print("--- Ngưỡng cố định áp cho cả ba mức nhiễu (rho = %.1f) ---" % RHO)
    a0, b0 = MUC["vừa"]
    ref = gamma_gamma(a0, b0, 200000)
    t_len, t_xuong = np.percentile(ref, 97.7), np.percentile(ref, 15.9)
    print("Ngưỡng: đơn", round(T_DON, 2), "| lên", round(t_len, 2), "| xuống", round(t_xuong, 2))
    for ten in MUC:
        tap = tao_tap_muc(ten, 100)
        _, doi_don, sai_don = chi_phi_goc(tap, T_DON, T_DON)
        _, doi_hys, sai_hys = chi_phi_goc(tap, t_len, t_xuong)
        print(ten, "| đổi (đơn/hys):", round(doi_don, 1), "/", round(doi_hys, 1),
              "| sai chế độ % (đơn/hys):", round(100 * sai_don), "/", round(100 * sai_hys))


# =====================================================================
# KHOI 10: MAIN - bat/tat tung giai doan
# Mac dinh dung CAP_CHOT, khong tim lai (KHOI 5 chay lau). Dat "tim_cap": True de tim lai
# tren may ban; khi do cap tim duoc duoc IN RA va dung cho cac giai doan sau (CAP_CHOT trong file
# khong bi sua - muon giu cap moi thi chep vao KHOI 0).
# =====================================================================
CHAY = {
    "demo": True,            # KHOI 9: demo minh hoa (nhanh)
    "tim_cap": False,        # KHOI 5: tim lai cap nguong (LAU: vai phut)
    "danh_gia": True,        # KHOI 6: danh gia cap chot tren chuoi moi (vai chuc giay)
    "lut": True,             # KHOI 7: LUT hai muc, so sanh voi Python, ghi lut.h (nhanh)
    "si": True,              # KHOI 8: chon nhom bang SI (kha nhanh)
}

if __name__ == "__main__":
    cap = dict(CAP_CHOT)
    cac_cap_moi = None
    if CHAY["demo"]:
        np.random.seed(SEED)
        demo_co_ban()
        np.random.seed(SEED)
        demo_nguong_co_dinh()
    if CHAY["tim_cap"]:
        np.random.seed(SEED)
        cac_cap_moi = tim_cap_cac_nhom()
        cap = {nhom: cac_cap_moi[(nhom, W_CHON)] for nhom in NHOM}
    if CHAY["danh_gia"]:
        np.random.seed(SEED)
        danh_gia_ket_qua(cap, cac_cap_moi)
    if CHAY["lut"]:
        np.random.seed(SEED)
        luts = tao_lut_hai_muc(cap)
        so_sanh_lut(luts, cap)
        ghi_lut_h(luts, cap)
    if CHAY["si"]:
        np.random.seed(SEED)
        phan_loai_si()
    print("--- Cặp ngưỡng đang dùng ---")
    for nhom in NHOM:
        print(nhom, "| lên =", round(cap[nhom][0], 2), "xuống =", round(cap[nhom][1], 2))
    print("Xong toàn bộ")
