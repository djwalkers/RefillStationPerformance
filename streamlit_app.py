elif active_tab == "High Performers":
    st.header("High Performers")

    # ---- Month and Day Filters ----
    month_options = sorted(data['Date'].dt.strftime('%B').dropna().unique())
    month_sel = st.selectbox("Filter by Month:", ["All"] + month_options, key="summary_month")
    filtered_data = data.copy()
    if month_sel != "All":
        filtered_data = filtered_data[filtered_data['Date'].dt.strftime('%B') == month_sel]

    day_options = sorted(filtered_data['Date'].dt.strftime('%d-%m-%Y').dropna().unique())
    day_sel = st.selectbox("Filter by Day:", ["All"] + day_options, key="summary_day")
    if day_sel != "All":
        filtered_data = filtered_data[filtered_data['Date'].dt.strftime('%d-%m-%Y') == day_sel]

    # Assign shift based on time
    def assign_shift(time_str):
        if pd.isna(time_str):
            return "Unknown"
        try:
            hour, minute = map(int, str(time_str).split(":"))
        except:
            return "Unknown"
        total_minutes = hour * 60 + minute
        if 6*60 <= total_minutes <= 14*60:
            return "AM"
        elif 14*60 < total_minutes <= 22*60:
            return "PM"
        else:
            return "Night"
    filtered_data['Shift'] = filtered_data['Time'].apply(assign_shift)

    def ensure_shift_columns(df, index_col="Date"):
        for shift in ["AM", "PM", "Night"]:
            if shift not in df.columns:
                df[shift] = 0
        cols = [index_col, "AM", "PM", "Night"]
        df = df[[c for c in cols if c in df.columns]]
        for shift in ["AM", "PM", "Night"]:
            if shift in df.columns:
                df[shift] = df[shift].fillna(0)
        return df

    trophy = "🏆 "

    # --- Top Picker Per Day (Carts Counted Per Hour) with Station Type ---
    daily_totals = (
        filtered_data.groupby(['Date', 'Users'], as_index=False)['Carts Counted Per Hour'].sum()
    )
    idx = daily_totals.groupby('Date')['Carts Counted Per Hour'].idxmax()
    top_picker_per_day = daily_totals.loc[idx].reset_index(drop=True)
    top_picker_per_day = top_picker_per_day.rename(
        columns={'Users': 'Top Picker', 'Carts Counted Per Hour': 'Total Carts Counted Per Hour'}
    )
    # Optional: Get their main station type for the day (station with their highest carts counted)
    main_station = (
        filtered_data.groupby(['Date', 'Top Picker', 'Station Type'], as_index=False)['Carts Counted Per Hour'].sum()
    )
    main_station_idx = main_station.groupby(['Date', 'Top Picker'])['Carts Counted Per Hour'].idxmax()
    main_station = main_station.loc[main_station_idx][['Date', 'Top Picker', 'Station Type']]
    top_picker_per_day = top_picker_per_day.merge(
        main_station,
        left_on=['Date', 'Top Picker'],
        right_on=['Date', 'Top Picker'],
        how='left'
    )

    if not top_picker_per_day.empty:
        top_picker_per_day['Date'] = pd.to_datetime(top_picker_per_day['Date']).dt.strftime('%d-%m-%Y')
        top_picker_per_day['Top Picker'] = trophy + top_picker_per_day['Top Picker'].astype(str)
        top_picker_per_day['Total Carts Counted Per Hour'] = top_picker_per_day['Total Carts Counted Per Hour'].apply(
            lambda x: f"{x:.2f}" if 0 < x < 1 else f"{int(round(x))}"
        )
    st.subheader("Top Picker Per Day (All Hours, Calendar Day)")
    st.markdown(
        "*Note: This table sums all picks by each user within the full calendar day, regardless of shift boundaries. "
        "A user’s total may differ from the sum of their per-shift totals if their activity crosses shift times.*",
        unsafe_allow_html=True,
    )
    # ---- Drilldown Implementation ----
    # Add a unique row number index for display selection
    top_picker_per_day_display = top_picker_per_day[['Date', 'Top Picker', 'Station Type', 'Total Carts Counted Per Hour']].reset_index(drop=True)
    st.dataframe(
        top_picker_per_day_display,
        use_container_width=True,
        hide_index=True,
        selection_mode="single",
        key="top_picker_per_day_df"
    )
    selected_row = st.session_state.get("top_picker_per_day_df_selected_rows", [])
    if selected_row:
        row = top_picker_per_day_display.iloc[selected_row[0]]
        selected_date = row["Date"]
        selected_user = row["Top Picker"].replace(trophy, "")  # Remove trophy emoji

        # Pull raw details for this user/date from your main DataFrame
        detail_df = data[
            (data["Date"].dt.strftime('%d-%m-%Y') == selected_date) &
            (data["Users"] == selected_user)
        ].copy()
        if not detail_df.empty:
            detail_df['Date'] = pd.to_datetime(detail_df['Date']).dt.strftime('%d-%m-%Y')
            detail_df['Carts Counted Per Hour'] = detail_df['Carts Counted Per Hour'].apply(
                lambda x: f"{x:.2f}" if 0 < x < 1 else f"{int(round(x))}"
            )
            detail_df = detail_df[[
                "Date", "Users", "Station Type", "Shift", "Time",
                "Drawers Counted", "Damaged Drawers Processed", "Damaged Products Processed",
                "Rogues Processed", "Carts Counted Per Hour"
            ]]
            st.markdown(f"#### Drilldown: Details for {selected_user} on {selected_date}")
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.info("No additional details available for this selection.")

    # --- (Continue with the rest of your tables as before...) ---
    # ... (Top Picker Per Shift, Carts Per Shift, Breakdown by Station Type/Shift, etc.)
