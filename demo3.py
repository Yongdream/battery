import matplotlib.pyplot as plt

# Designing the common-emitter amplifier circuit
# Parameters
Vcc = 12  # Supply voltage
Rc = 2.2e3  # Collector resistor in ohms
Re = 1e3  # Emitter resistor in ohms
Beta = 100  # Current gain of the transistor
Vbe = 0.7  # Base-emitter voltage in volts
Ib = 0.1e-3  # Base current in amps
Ic = Beta * Ib  # Collector current in amps
Vce = Vcc - Ic * Rc  # Collector-emitter voltage in volts

# Components
components = {
    "Vcc": Vcc,
    "Rc": Rc,
    "Re": Re,
    "Beta": Beta,
    "Vbe": Vbe,
    "Ib": Ib,
    "Ic": Ic,
    "Vce": Vce
}

# Plotting the schematic
fig, ax = plt.subplots()
ax.text(0.2, 0.9, 'Vcc', fontsize=14, verticalalignment='center')
ax.plot([0.5, 0.5], [0.8, 0.2], color='black')
ax.text(0.5, 0.2, 'Rc', fontsize=14, verticalalignment='center')
ax.plot([0.5, 0.5], [0.2, 0.1], color='black')
ax.plot([0.5, 0.3], [0.1, 0.1], color='black')
ax.text(0.3, 0.1, 'Q', fontsize=14, verticalalignment='center')
ax.plot([0.3, 0.3], [0.1, -0.1], color='black')
ax.plot([0.3, 0.5], [-0.1, -0.1], color='black')
ax.text(0.5, -0.1, 'Re', fontsize=14, verticalalignment='center')
ax.plot([0.5, 0.5], [-0.1, -0.2], color='black')
ax.text(0.5, -0.2, 'Ground', fontsize=14, verticalalignment='center')

ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 1)
ax.axis('off')
plt.title("Common-Emitter Amplifier Schematic")
plt.show()

