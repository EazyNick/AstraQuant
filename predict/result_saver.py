"""
결과 저장 관련 클래스
"""

import pandas as pd
import numpy as np


class ResultSaver:
    """예측 결과 저장 클래스"""
    
    def __init__(self, log_manager):
        self.log_manager = log_manager
        
    def save_prediction_results(self, predictions, probs_list):
        """예측 결과를 CSV 파일로 저장"""
        # 전체 액션 확률을 저장용 리스트로 정리
        all_prob_records = []
        for i in range(len(predictions)):
            date = predictions[i][0]
            main_action = predictions[i][1]
            main_prob = predictions[i][2]
            prob_row = probs_list[i]  # 확률 분포

            row = {
                "날짜": date,
                "예측 매매 결정": main_action,
                "확률(%)": main_prob,
            }
            for j in range(len(prob_row)):
                row[f"Action_{j}_확률(%)"] = prob_row[j]
            all_prob_records.append(row)

        # 데이터프레임으로 변환
        detailed_result_df = pd.DataFrame(all_prob_records)

        # CSV로 저장
        output_csv_path = "output/prediction_result_detailed.csv"
        detailed_result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        self.log_manager.logger.info(f"📁 예측 결과가 CSV로 저장되었습니다: {output_csv_path}")
        
        return detailed_result_df
        
    def log_prediction_summary(self, result_df, probs):
        """예측 결과 요약 로그 출력"""
        # 데이터프레임 출력 설정
        pd.set_option("display.max_rows", None)
        
        self.log_manager.logger.info("📌 예측 결과 (상위 5개)")
        self.log_manager.logger.info(result_df.head(5))

        self.log_manager.logger.info("📌 예측 결과 (하위 5개)")
        self.log_manager.logger.info(result_df.tail(5))

        # 각 매매 결정별 총 개수 계산
        total_sell = result_df["예측 매매 결정"].str.startswith("매도").sum()
        total_hold = result_df["예측 매매 결정"].str.startswith("관망").sum()
        total_buy = result_df["예측 매매 결정"].str.startswith("매수").sum()

        summary = f"총 매도: {total_sell}건, 총 관망: {total_hold}건, 총 매수: {total_buy}건"
        self.log_manager.logger.info(summary)

        # 확률 분포에서 매수/매도/관망 각각의 총합 계산
        # probs는 이미 1차원 배열이므로 직접 접근
        buy_prob_sum = np.sum(probs[1:2]) if len(probs) > 1 else 0
        sell_prob_sum = np.sum(probs[2:3]) if len(probs) > 2 else 0
        hold_prob = probs[0] if len(probs) > 0 else 0

        total_sum = buy_prob_sum + sell_prob_sum + hold_prob

        if total_sum > 0:
            buy_percent = (buy_prob_sum / total_sum) * 100
            sell_percent = (sell_prob_sum / total_sum) * 100
            hold_percent = (hold_prob / total_sum) * 100

            self.log_manager.logger.info(f"📊 전체 확률 분포 요약:")
            self.log_manager.logger.info(f"🟩 매수(Buy) 확률 총합: {buy_prob_sum:.2f} ({buy_percent:.2f}%)")
            self.log_manager.logger.info(f"🟥 매도(Sell) 확률 총합: {sell_prob_sum:.2f} ({sell_percent:.2f}%)")
            self.log_manager.logger.info(f"🟨 관망(Hold) 확률: {hold_prob:.2f} ({hold_percent:.2f}%)")
        else:
            self.log_manager.logger.warning("⚠️ 확률 데이터가 올바르지 않습니다.")
        
    def log_final_predictions(self, predictions, probs, action, df):
        """최종 예측 결과 로그 출력"""
        # 실제 데이터의 마지막 날짜 (df 기준)
        true_last_date = df.iloc[-1]['Date']

        # 예측 구간 기준 마지막 날짜 및 액션 결과
        predicted_last_date, last_action_str, _ = predictions[-1]
        last_action_index = action  # 마지막 루프에서 나온 action 값
        
        # probs는 이미 1차원 배열이므로 직접 접근
        if len(probs) > last_action_index:
            last_action_prob = probs[last_action_index]
        else:
            last_action_prob = 0.0

        self.log_manager.logger.info(f"📅 예측 가능한 구간의 마지막 날짜: {predicted_last_date}")
        self.log_manager.logger.info(f"📈 마지막 예측 액션: {last_action_str} (확률: {last_action_prob:.2f}%)")

        # 액션 유형별 상세 로그
        if last_action_str.startswith("매수"):
            if "주" in last_action_str:
                shares_bought = int(last_action_str.split()[1].replace("주", ""))
                self.log_manager.logger.info(f"🛒 마지막 시점에서 {shares_bought}주 매수 예정 (확률: {last_action_prob:.2f}%)")
            else:
                self.log_manager.logger.info(f"🛒 마지막 시점에서 전량 매수 예정 (확률: {last_action_prob:.2f}%)")
        elif last_action_str.startswith("매도"):
            if "주" in last_action_str:
                shares_sold = int(last_action_str.split()[1].replace("주", ""))
                self.log_manager.logger.info(f"💰 마지막 시점에서 {shares_sold}주 매도 예정 (확률: {last_action_prob:.2f}%)")
            else:
                self.log_manager.logger.info(f"💰 마지막 시점에서 전량 매도 예정 (확률: {last_action_prob:.2f}%)")
        else:
            self.log_manager.logger.info(f"⏸ 마지막 시점에서는 관망(Hold) 상태입니다. (확률: {last_action_prob:.2f}%)") 