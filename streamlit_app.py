def show_bar_chart(df, x, y, title):
    total = df[x].sum()
    st.markdown(
        f"<div style='color:{FG_COLOR}; font-size:18px; font-weight:bold; margin-bottom:10px'>Total {x.replace('_',' ')}: {int(total):,}</div>",
        unsafe_allow_html=True,
    )
    if df.empty:
        st.info("No data to display for this selection.")
        return
    fig, ax = plt.subplots(figsize=(max(7, len(df) * 0.45), 6))
    bars = ax.bar(df[y], df[x], color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=2)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(round(height))}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, color=FG_COLOR, fontweight="bold")
    ax.set_ylabel(x.replace('_', ' '), color=FG_COLOR, weight="bold")
    ax.set_xlabel(y.replace('_', ' '), color=FG_COLOR, weight="bold")
    ax.set_title(title, color=FG_COLOR, weight="bold")
    ax.tick_params(axis='x', colors=FG_COLOR, rotation=35)
    ax.tick_params(axis='y', colors=FG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    plt.tight_layout()
    st.pyplot(fig)
