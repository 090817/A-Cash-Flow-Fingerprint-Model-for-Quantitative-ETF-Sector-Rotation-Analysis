# 方法1: 直接运行完整分析
python etf_sector_recommendation.py

# 方法2: 在代码中使用
from etf_sector_recommendation import ETFSectorRecommendationSystem

# 创建系统实例
system = ETFSectorRecommendationSystem()

# 运行分析
system.run_analysis()

# 显示结果
system.display_results()

# 保存结果
system.save_results()

# 获取特定行业信息
tech_info = system.get_sector_details("科技")