"""
Hybrid 병합 스크립트 (EASE + VAE)

실행 순서:
1. multi_ease_candidates/merged_candidates.parquet 로드
2. vae_candidates.parquet 로드
3. Outer Join + Cross-Scoring
4. hybrid_candidates.parquet 저장

입력:
- multi_ease_candidates/merged_candidates.parquet
- vae_candidates.parquet

출력:
- hybrid_candidates.parquet
"""
from src.mergers.hybrid_merger import merge_hybrid_candidates


def main():
    """Hybrid 병합 메인 함수."""
    print("\n🔥 Hybrid 병합 시작 (EASE + VAE)")
    merge_hybrid_candidates()
    print("✅ Hybrid 병합 완료!")


if __name__ == "__main__":
    main()

