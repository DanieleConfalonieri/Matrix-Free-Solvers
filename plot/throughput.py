import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("tscal_ncore_p6_m3.csv")
df['Throughput'] = df['Throughput(MDoFs/s)'].str.replace(' MDoFs/s', '').astype(float)

df_56 = df[df['Cores'] == 56]
df_p6 = df[df['Degree'] == 6]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Throughput vs Degree
ax1.plot(df_56[df_56['Solver']=='MatrixFree']['Degree'], df_56[df_56['Solver']=='MatrixFree']['Throughput'], 'o-', label='Matrix-Free', color='#1f77b4', linewidth=2)
ax1.plot(df_56[df_56['Solver']=='MatrixBased']['Degree'], df_56[df_56['Solver']=='MatrixBased']['Throughput'], 's--', label='Matrix-Based', color='#d62728', linewidth=2)
ax1.set_xlabel('Polynomial Degree $p$')
ax1.set_ylabel('Throughput (MDoFs/s)')
ax1.set_title('Throughput vs Polynomial Degree (56 Cores)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Throughput vs Cores
ax2.plot(df_p6[df_p6['Solver']=='MatrixFree']['Cores'], df_p6[df_p6['Solver']=='MatrixFree']['Throughput'], 'o-', label='Matrix-Free', color='#1f77b4', linewidth=2)
ax2.plot(df_p6[df_p6['Solver']=='MatrixBased']['Cores'], df_p6[df_p6['Solver']=='MatrixBased']['Throughput'], 's--', label='Matrix-Based', color='#d62728', linewidth=2)
ax2.set_xlabel('Number of Cores')
ax2.set_ylabel('Throughput (MDoFs/s)')
ax2.set_title('Throughput Scaling (Degree $p=6$)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('throughput_combined.pdf', bbox_inches='tight')